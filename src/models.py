from typing import Union
import torch
import torchmetrics
import torchmetrics.classification as tmc
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import wandb
import deep_learning.src.utils as utils


class SegmentationModel(pl.LightningModule):
    def __init__(self, 
        model: str = "Unet",
        backbone: str = "resnet18",
        loss: str = "dice",
        loss_smooth: Union[float, None] = 0.0,
        num_channels: int = 3,
        num_classes: int = 1,
        weights: Union[str, None] = None,
        use_batchnorm:  Union[bool, None] = True,
        pooling: Union[str, None] = "avg",
        dropout: Union[float, None] = 0.0,    
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
        lr_scheduler: Union[str, None] = None,
        weight_decay: Union[float, None] = 0.0,
        amsgrad: Union[bool, None] = False,
        momentum: Union[float, None] = 0.0,
        gamma: Union[float, None] = 0.1,
        step_size: Union[int, None] = 10,
        T_0: Union[int, None] = 5,
        T_mult: Union[int, None] = 1,
        patience: Union[int, None] = 5,
        min_lr: Union[float, None] = 1e-8,
        cooldown: Union[int, None] = 0,
        freeze_backbone: bool = False,
        result_dir: Union[str, None] = None,
    ) -> None:
        super().__init__()
        self.result_dir = result_dir
        self.save_hyperparameters(ignore=["result_dir"])

        if self.hparams.model == 'DeepLabV3Plus':
            model_load = getattr(smp, self.hparams.model)(
                encoder_name=self.hparams.backbone,
                encoder_weights=self.hparams.weights,
                in_channels=self.hparams.num_channels,
                classes=self.hparams.num_classes,
                )
        else:
            model_load = getattr(smp, self.hparams.model)(
                encoder_name=self.hparams.backbone,
                encoder_weights=self.hparams.weights,
                decoder_use_batchnorm=self.hparams.use_batchnorm,
                in_channels=self.hparams.num_channels,
                classes=self.hparams.num_classes,
                )

        self.model = model_load
        if self.hparams.freeze_backbone:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        
        loss_kwargs = {
            "DiceLoss": {"mode": smp.losses.BINARY_MODE, 
                         "from_logits": True, 
                         "smooth": self.hparams.loss_smooth}
                         } # more losses can be added

        self.loss_fn = getattr(smp.losses, self.hparams.loss)(
            **loss_kwargs.get(loss, {})
        )

        metrics = torchmetrics.MetricCollection({
                "iou": tmc.BinaryJaccardIndex(
                    threshold=0.5,
                ),
                "ap": tmc.BinaryAveragePrecision(
                    thresholds=None
                ),
                "precision": tmc.BinaryPrecision(
                    threshold=0.5,
                    multidim_average="global",
                ),
                "recall": tmc.BinaryRecall(
                    threshold=0.5,
                    multidim_average="global",
                ),
                "f1": tmc.BinaryF1Score(
                    threshold=0.5,
                    multidim_average="global",
                )})
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.acc_xs =  []
        self.acc_ys = []
        self.acc_y_hats = []
        self.count = 0

    def configure_optimizers(self):
        optimizer_kwargs = {
            "Adam": {"weight_decay": self.hparams.weight_decay, "amsgrad": self.hparams.amsgrad},
            "AdamW": {"weight_decay": self.hparams.weight_decay, "amsgrad": self.hparams.amsgrad},
            "SGD": {"momentum": self.hparams.momentum}
        }
        optimizer_args = {
            "lr": self.hparams.learning_rate,
        }
        optimizer_args.update(optimizer_kwargs.get(self.hparams.optimizer, {}))
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.model.parameters(), 
            **optimizer_args
        )

        if self.hparams.lr_scheduler is None:
            return optimizer

        lr_scheduler_kwargs = {
            "ExponentialLR": {"gamma": self.hparams.gamma},
            "StepLR": {"gamma": self.hparams.gamma, "step_size": self.hparams.step_size},
            "CosineAnnealingWarmRestarts": {"T_0": self.hparams.T_0, "eta_min": self.hparams.min_lr, "T_mult": self.hparams.T_mult},
            "ReduceLROnPlateau": {"patience": self.hparams.patience, 
                                "min_lr": self.hparams.min_lr, 
                                "cooldown": self.hparams.cooldown,
                                "mode": "max"},
        }
        lr_scheduler_args = lr_scheduler_kwargs.get(self.hparams.lr_scheduler, {})

        lr_scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler)(
                    optimizer, 
                    **lr_scheduler_args
                )
        if self.hparams.lr_scheduler == "ReduceLROnPlateau":
            return {"optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "monitor": "val_iou",
                        "interval": "epoch",
                        "frequency": 1
                    }}
        else:
            return [optimizer], [lr_scheduler]      

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch, batch_idx):   
        x, y = batch["image"], batch["mask"]
        y_hat = self.forward(x)
        return x, y, y_hat

    def training_step(self, batch, batch_idx):
        _, y, y_hat = self._shared_step(batch, batch_idx)
        loss = self.loss_fn(y_hat.squeeze(1), y.float())
        y_iou, y_hat_iou = utils.get_data_to_log(y, y_hat)
        if y_iou is not None:
            self.train_metrics.update(y_hat_iou.squeeze(1), y_iou)    
        self.log("train_loss", loss, on_step=True, on_epoch=True)        
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric("val_iou", summary="max")
        x, y, y_hat = self._shared_step(batch, batch_idx)
        y_hat_hard = utils.apply_sigmoid(y_hat) 
        loss = self.loss_fn(y_hat.squeeze(1), y.float())
        y_iou, y_hat_iou = utils.get_data_to_log(y, y_hat)
        if y_iou is not None:
            self.val_metrics.update(y_hat_iou.squeeze(1), y_iou)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.acc_xs.append(x)
        self.acc_ys.append(y)
        self.acc_y_hats.append(y_hat_hard.squeeze(1))
        return y_hat

    def on_validation_epoch_end(self):
        x = torch.cat(self.acc_xs)
        y = torch.cat(self.acc_ys)
        y_hat = torch.cat(self.acc_y_hats)
        log_image = True
        count_log = 0
        for i in range(y.size(0)):
            if torch.any(y[i] > 0) and log_image:
                image = utils.prepare_sample_images(x[i][0], y[i], (y_hat[i] >= 0.5).int())
                self.logger.log_image(key='val_samples', images=[image], step=self.current_epoch)
                if count_log > 2:
                    log_image = False 
                count_log += 1
        confmat_img = utils.compute_confusion_matrix(y_hat=y_hat, y=y)
        self.logger.log_image(key="val_confusion_matrix", images=[confmat_img], step=self.current_epoch)
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()
        self.acc_xs.clear()
        self.acc_ys.clear()
        self.acc_y_hats.clear()
    
    def test_step(self, batch, batch_idx):
        _, y, y_hat = self._shared_step(batch, batch_idx)
        y_hat_hard = utils.apply_sigmoid(y_hat)
        y_iou, y_hat_iou = utils.get_data_to_log(y, y_hat)
        if y_iou is not None:
            self.test_metrics.update(y_hat_iou.squeeze(1), y_iou)
        self.acc_ys.append(y)
        self.acc_y_hats.append(y_hat_hard.squeeze(1))
        self.count = utils.save_results(self.count, y_hat_hard, batch, self.result_dir, stage='test')
        

    def on_test_epoch_end(self):
        y = torch.cat(self.acc_ys)
        y_hat_hard = torch.cat(self.acc_y_hats)
        confmat_img = utils.compute_confusion_matrix(y_hat=y_hat_hard, y=y)
        self.logger.log_image(key="test_confusion_matrix", images=[confmat_img], step=self.current_epoch)
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
        self.acc_ys.clear()
        self.acc_y_hats.clear()

    def predict_step(self, batch, batch_idx):
        _, _, y_hat = self._shared_step(batch, batch_idx)
        y_hat_hard = utils.apply_sigmoid(y_hat) 
        self.count = utils.save_results(self.count, y_hat_hard, batch, self.result_dir, stage='predict')

