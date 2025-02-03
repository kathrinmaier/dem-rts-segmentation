import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import rasterio
import torch
import torchvision
import torch.nn as nn
import torchmetrics.classification as tmc
import wandb

def apply_sigmoid(y_hat):
    m = nn.Sigmoid()
    y_hat_hard = m(y_hat) 
    return y_hat_hard

def get_data_to_log(y, y_hat):
    new_y = []
    new_y_hat = []
    count = 0
    for i in range(y.size(0)):
        if torch.any(y[i] > 0):
            new_y.append(y[i])
            new_y_hat.append(y_hat[i])
            count += 1
    if not count == 0:
        y_iou = torch.stack(new_y)
        y_hat_iou = torch.stack(new_y_hat)
        return y_iou, y_hat_iou
    else:
        return None, None

def prepare_samples_to_log(x, y, y_hat_int):
    mask = y.unsqueeze(0).expand(3, x.shape[1], x.shape[2])
    pred_int = torch.cat([y_hat_int[0].unsqueeze(0), y_hat_int[0].unsqueeze(0), y_hat_int[0].unsqueeze(0)])
    images = torchvision.utils.make_grid([x, mask, pred_int], nrow=2, pad_value=4)
    return images

def prepare_sample_images(x, y, y_hat_int):
    x, y, y_hat_int = x.cpu(), y.cpu(), y_hat_int.cpu()
    images = torchvision.utils.make_grid([x, y, y_hat_int], nrow=1, pad_value=4)
    fig, axs = plt.subplots(ncols=len(images), squeeze=False)
    axs[0, 0].imshow(images[0], cmap='bwr', vmax=3.0, vmin=-3.0)
    axs[0, 0].set_title("DEM")
    axs[0, 1].imshow(images[1], cmap='Blues')
    axs[0, 1].set_title("Ground truth")
    axs[0, 2].imshow(images[2], cmap='Reds')
    axs[0, 2].set_title("Prediction")
    for i in range(0,3):
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.close()
    return fig

def compute_confusion_matrix(y_hat, y):
        confmat = tmc.BinaryConfusionMatrix(threshold=0.5, normalize='true').cpu()
        confmat(y_hat.cpu(), y.cpu().int())
        confmat_comp = confmat.compute()
        df_cm = pd.DataFrame(confmat_comp)
        plt.figure(figsize = (7,7))
        confmat_img = sns.heatmap(df_cm, annot=True, fmt=".3f", cmap="bwr").get_figure()
        plt.close(confmat_img)
        return confmat_img

def prepare_segmentation_mask(x, y, y_hat_hard, epoch):
    class_labels = {0: "background", 1: "RTS"}
    image = x.cpu().numpy()
    mask = y.cpu().numpy()
    prediction = y_hat_hard.cpu().numpy()
    mask_img = wandb.Image(image, masks={
        "prediction":{
            "mask_data": prediction,
            "class_labels": class_labels
        },
        "ground_truth":{
            "mask_data": mask,
            "class_labels": class_labels
        }})
    data = [epoch, mask_img]
    return data


def save_results(count, y_hat_hard, batch, result_dir, stage):
    os.makedirs(result_dir, exist_ok=True)
    for i in range(y_hat_hard.size(0)):
        count +=1
        result_file = f"{result_dir}/test_{count}.tif"
        y_hat_array = y_hat_hard[i].cpu().numpy()
        bbox = batch['bbox'][i]
        if stage == 'test':
            width = y_hat_array.shape[2]
            height = y_hat_array.shape[1]
            meta = {
                "driver": "GTiff",
                "count": 1,
                "height": height,
                "width": width,
                "crs": batch["crs"][i],
                "transform": rasterio.transform.from_bounds(west=bbox[0].item(), 
                                                            south=bbox[2].item(), 
                                                            east=bbox[1].item(), 
                                                            north=bbox[3].item(), 
                                                            width=width, 
                                                            height=height),
                "dtype": y_hat_array.dtype,
            }
            with rasterio.open(result_file, 'w', **meta) as dst:
                dst.write(y_hat_array)
        elif stage == 'predict':
            clip_margin = 0 # 28
            clip_size = 0 # 280
            width = y_hat_array.shape[2]-(2*clip_margin)
            height = y_hat_array.shape[1]-(2*clip_margin)
            meta = {
                "driver": "GTiff",
                "count": 1,
                "height": height,
                "width": width,
                "crs": batch["crs"][i],
                "transform": rasterio.transform.from_bounds(west=bbox[0].item()+clip_size, 
                                                            south=bbox[2].item()+clip_size, 
                                                            east=bbox[1].item()-clip_size, 
                                                            north=bbox[3].item()-clip_size, 
                                                            width=width, 
                                                            height=height),
                "dtype": y_hat_array.dtype,
            }
            with rasterio.open(result_file, 'w', **meta) as dst:
                #dst.write(y_hat_array[:, clip_margin:-clip_margin, clip_margin:-clip_margin])
                dst.write(y_hat_array[:, :, :])

    return count
                