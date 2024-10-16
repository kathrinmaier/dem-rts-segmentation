import os
import argparse
import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from deep_learning.src.prepare_data import select_data
from deep_learning.src.models import SegmentationModel
import deep_learning.src.transforms as T
from deep_learning.src.datasets import RTSDetectionDataModule


def main(cfg, mode):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using", device, "device")
    data_paths = select_data(cfg, mode)
    pl.seed_everything(42, workers=True)
    if cfg.data.augmentation:
        augmentations = T.augmentations()
    else:
        augmentations = None

    if cfg.model_ckpt.use_ckpt:
        ckpt_path = f"{cfg.paths.root}/{cfg.model_ckpt.ckpt_path}/{cfg.model_ckpt.run_id}/checkpoints"
        ckpt_file = os.listdir(ckpt_path)[0]
        result_dir = f"{cfg.paths.root}_{cfg.model_ckpt.run_id}"
        os.makedirs(result_dir, exist_ok=True)
        model = SegmentationModel.load_from_checkpoint(f"{ckpt_path}/{ckpt_file}", result_dir=result_dir)
    else:
        model = SegmentationModel(**cfg.module, result_dir=None)

    datamodule = RTSDetectionDataModule(data_paths=data_paths, augmentations=augmentations, **cfg.datamodule)

    lr_monitor = LearningRateMonitor(**cfg.callbacks.lr_monitor)
    early_stop_callback = EarlyStopping(**cfg.callbacks.earlystop)
    checkpoint_callback = ModelCheckpoint(**cfg.callbacks.checkpoint)
    train_callbacks = [lr_monitor, early_stop_callback, checkpoint_callback]
    logger = WandbLogger(**cfg.logging)
    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(logger=logger, callbacks=train_callbacks, **cfg.trainer)

    if mode == "fit":
        trainer.fit(model=model, datamodule=datamodule)
        
    if mode == "test":
        trainer.test(model=model, datamodule=datamodule)
    
    if mode == "predict":
        trainer.predict(model=model, datamodule=datamodule)

    wandb.finish()
    return





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", 
        type=str, 
        default='deep_learning/config/config.yml',
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fit",
        choices=["fit", "test", "predict"],
        help="Choice of mode",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    main(cfg, args.mode)