# dem-rts-segmentation [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This is the repository to the publication "Detecting Mass Wasting of Retrogressive Thaw Slumps in Spaceborne Elevation Models using Deep Learning" by Kathrin Maier, Philipp Bernhard, Sophia Ly, Michele Volpi, Ingmar Nitze, Shiyi Li, and Irena Hajnsek.


- [dem-rts-segmentation ](#dem-rts-segmentation-)
  - [General](#general)
  - [Dataset](#dataset)
  - [Usage](#usage)
    - [Training / Testing / Inference](#training--testing--inference)
    - [Postprocessing](#postprocessing)
    - [Performance assessment](#performance-assessment)
  - [Authors](#authors)

## General
We use binary semantic segmentation based on commonly used Deep Learning models (UNet, UNet++, and DeepLabV3+ from the Python libray [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)) to detect mass wasting from Retregressive Thaw Slumps (RTSs) on difference images of time-series Digital Elevation Models (DEM). We implemented our pipeline based on [Pytorch](https://pytorch.org/).

We generated the DEMs with bistatic InSAR satellite observations from the TanDEM-X mission. The data can be requested through scientific research proposals, and the processing pipeline follows standard InSAR methods using the Gamma Remote Sensing Software as described in the publication. The code for generating DEMs from TanDEM-X SAR observations can be made available upon request. In this repository, we provide a small example of the data we have used to train and test the deep learning models. 

## Dataset
The dataset produced by this code and described in the associated journal article can be found under [DOI:10.3929](https://doi.org/10.3929/ethz-b-000718475) and contains the extents of the training, testing, and inference study sites as well as training labels and the predicted RTS polygons including mass wasting information.


## Usage

### Training / Testing / Inference
- main.py should be called from the console with implemented CGI as follows: 
```-- mode predict # specify train, test, or predict```
```-- config config/config.yml # specify config file```

- [config.yml](config/config.yml): stores all configuration parameters that can be adapted. 

- [prepare_data.py](src/prepare_data.py): prepares data for [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) and [TorchGeo](https://github.com/microsoft/torchgeo). Needs to be adapted to specific needs.

- [datasets.py](src/datasets.py): [TorchGeo](https://github.com/microsoft/torchgeo) handles the geodata part of the pipeline. We have adopted parts of the code to our specific dataset (no optical data, four input channels)

- [transforms.py](src/transforms.py): Sequential augmentations including horizontal and vertical flip, affine transform and Gaussian blur with certain probability to be executed to the input data on the fly.

- [models.py](src/models.py): [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) for UNet, UNet++, and DeepLabV3+ architectures and all available encoders. Optimisers can be chosen from Adam, AdamW, and SGD. Set of learning rate schedulers to choose from (ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR, ExponentialLR). Logging of confusion matrix, and example prediction images to specific folder in [config](config/config.yml). 

### Postprocessing
We implemented a [postprocessing](postprocess) routine to decrease the number of false positive predictions common to unbalanced segmentation tasks. The approach includes
- **Size thresholding**: Exclude predictions smaller than 1000 m$^2$.
- **Elevation change thresholding**: Exclude prediction with minimum elevation changes smaller than 2 metres.
- **Water bodies**: Exclude predictions that intersect water bodies by more than 50% (based on an external binary water body mask, e.g. from NDWI from Sentinel-2 images).
- **SAR quality**: Exclude predictions that intersect with areas experiencing SAR quality issues (low coherence, layover/shadow, high local incidence angle) by more than 50%.
- **Mass wasting quantification**: The script calculates mass wasting attributes such as RTS area change, volume change, as well as the associated error bounds estimated from the coherence of the InSAR observations.

[postprocess.py](postprocess/postprocess.py): define starting and ending years of DEM difference images and coordinate system in [postprocess.ini](postprocess/postprocess.ini), as well as a ```ROOT```, ```TEMP```, and ```DATA``` path in the script, before running postprocess.py.

### Performance assessment
[compare.py](postprocess/compare.py) and [validation.py](postprocess/validation.py) are used to assess the accuracy of the predicted mass wasting quantities before and after postprocessing. Run [validation.py](postprocess/validation.py) on the reference labels first to add the mass wasting attributes. [compare.py](postprocess/compare.py) uses the predictions and reference labels with mass wasting attributes to calculate performance statistics.


## Authors
[@kathrinmaier](https://www.github.com/kathrinmaier), contact information: maierk@ethz.ch

