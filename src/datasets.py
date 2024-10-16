import re
import os
from typing import Any, cast, Union, Sequence, Optional
from collections.abc import Iterable
import numpy as np
import rasterio
import torch
from torch import Tensor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler
from einops import rearrange


class RTSDetectionDEMDataset(RasterDataset):
    filename_glob = f"norm_diff_dem_*.tif"
    diff_dem_values = [-10.0, 10.0]
    error_values = [0.0, 10.0]
    slope_values = [0.0, 90.0]
    filename_regex = r"(?P<band>[a-z]*_?[a-z]*_?[a-z]+).?.tif"
    separate_files = True
    all_bands = ["norm_diff_dem", "diff_dem", "diff_error", "slope"]

    def __getitem__(self, query) -> dict[str, Any]:
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])
        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )
        if self.separate_files:
            data_list: list[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in self.bands:
                band_filepaths = []
                for filepath in filepaths:
                    filename = os.path.basename(filepath)
                    directory = os.path.dirname(filepath)
                    match = re.match(filename_regex, filename)
                    if match:
                        if "band" in match.groupdict():
                            end = match.end("band")
                            filename = band + filename[end:]
  
                    filepath = os.path.join(directory, filename)
                    band_filepaths.append(filepath)
                data_list.append(self._merge_files(band_filepaths, query))
            data = torch.cat(data_list)

        else:
            data = self._merge_files(filepaths, query, self.band_indexes)
            
        query = torch.Tensor([query.minx, query.maxx, query.miny, query.maxy, query.mint, query.maxt])
        sample = {"crs": self.crs, "bbox": query}
        for i, (data_band, value_range) in enumerate(zip(data[1:4], [self.diff_dem_values, self.error_values, self.slope_values]), start=1):
            data[i] = 6 * ((data_band - value_range[0]) / (value_range[1] - value_range[0] + 1e-10)) - 3 if i < 3 else (data_band - value_range[0]) / (value_range[1] - value_range[0] + 1e-10)
            mask = torch.isnan(data[i]) | (data[i] > 10.0) | (data[i] < -10.0)
            data[i][mask] = 0
        data[torch.isnan(data)] = 0
        sample['image'] = data.float()
        return sample


    def _merge_files(
            self,
            filepaths,
            query,
            band_indexes: Optional[Sequence[int]] = None,
        ):
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

            bounds = (query.minx, query.miny, query.maxx, query.maxy)
            dest, _ = rasterio.merge.merge(vrt_fhs, bounds, self.res, method='first', nodata=0.0, indexes=band_indexes)
            if dest.dtype == np.uint16:
                dest = dest.astype(np.int32)
            elif dest.dtype == np.uint32:
                dest = dest.astype(np.int64)

            tensor = torch.tensor(dest)
            return tensor

    def _load_warp_file(self, filepath: str):
        src = rasterio.open(filepath, nodata=-9999)
        return src

class RTSDetectionMaskDataset(RasterDataset):
    filename_glob = f"mask*.tif"
    is_image = False
    seperate_files = False

    def __getitem__(self, query) -> dict[str, Any]:
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data = self._merge_files(filepaths, query, self.band_indexes)
        query = torch.Tensor([query.minx, query.maxx, query.miny, query.maxy, query.mint, query.maxt])
        sample = {"crs": self.crs, "bbox": query}
        data[data == 255] = 0
        sample["mask"] = data.squeeze().long()
        return sample


class RTSDetectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_paths: Union[str, Iterable[str], dict[str, Any]] = "data",
        batch_size: int = 8,
        num_workers: int = 1,
        img_size: int = 512,
        random: bool = False,
        trainset_length: int = 1000,
        augmentations = None,
    ) -> None:

        super().__init__()
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.random = random
        self.trainset_length = trainset_length
        self.augmentations = augmentations
 
    def setup(self, stage):
        if stage == 'fit':
            train_dem_paths = self.data_paths["train"][0]
            train_mask_paths = self.data_paths["train"][1]
            val_dem_paths = self.data_paths["val"][0]
            val_mask_paths = self.data_paths["val"][1]

            self.train_dataset = RTSDetectionDEMDataset(
                paths=train_dem_paths, 
                cache=False,
                ) & RTSDetectionMaskDataset(paths=train_mask_paths, cache=False)

            self.val_dataset = RTSDetectionDEMDataset(
                paths=val_dem_paths, 
                cache=False, 
                ) & RTSDetectionMaskDataset(paths=val_mask_paths, cache=False)

        elif stage == 'test':
            test_dem_paths = self.data_paths[0]
            test_mask_paths = self.data_paths[1]
            self.test_dataset = RTSDetectionDEMDataset(
                paths=test_dem_paths, 
                cache=False, 
                ) & RTSDetectionMaskDataset(paths=test_mask_paths, cache=False)

        elif stage == 'predict':
            self.predict_dataset = RTSDetectionDEMDataset(
                paths=self.data_paths, 
                cache=False, 
            )


    def train_dataloader(self):
        train_sampler = GridGeoSampler(
            dataset=self.train_dataset, 
            size=self.img_size, 
            stride=self.img_size-int(self.img_size/8),
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            collate_fn=stack_samples,
            num_workers=self.num_workers,
            drop_last=True,
        )
    
    def val_dataloader(self):
        val_sampler = GridGeoSampler(
            dataset=self.val_dataset, 
            size=self.img_size, 
            stride=self.img_size-int(self.img_size/8),
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            collate_fn=stack_samples,
            num_workers=self.num_workers,
            drop_last=True
            )

    def test_dataloader(self):
        test_sampler = GridGeoSampler(
            dataset=self.test_dataset,
            size=self.img_size, 
            stride=int(self.img_size-int(self.img_size/8)),
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=test_sampler,
            collate_fn=stack_samples,
        )

    def predict_dataloader(self):
        predict_sampler = GridGeoSampler(
            dataset=self.predict_dataset, 
            size=self.img_size, 
            stride=self.img_size-int(self.img_size/8),
            )
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=predict_sampler,
            collate_fn=stack_samples,
        )


    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            if self.augmentations is not None:
                batch["mask"] = rearrange(batch["mask"], "b h w -> b () h w")
                batch["mask"] = batch["mask"].to(torch.float)
                batch["image"], batch["mask"] = self.augmentations(
                    batch["image"], batch["mask"]
                )
                batch["mask"] = batch["mask"].to(torch.long)
                batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")
        return batch