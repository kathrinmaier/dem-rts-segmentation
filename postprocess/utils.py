import os
import ast
import numpy as np
import rasterio as rio
from rasterio.merge import merge
from rasterio.mask import mask
import shapely
from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely.geometry import Polygon
import geopandas as gpd
import fiona.transform
import pyproj
from rasterio.features import shapes
from configparser import ConfigParser


def get_config_dict(filename, section):
    parser = ConfigParser()
    parser.read(filename)
    config_dictionary = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config_dictionary[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return config_dictionary

def read_and_write_dem_data(diff_dem_objects, polygon, all_touched=False):
    diff_dems = []
    files = []
    errors = []
    counts = []
    for diff_dem_obj in diff_dem_objects:
        if isinstance(diff_dem_objects, list) and all(isinstance(item, dict) for item in diff_dem_objects):
            diff_dem_file = diff_dem_obj["diff_dem"]
            diff_error_file = diff_dem_obj["diff_error"]
        else:
            diff_dem_file = diff_dem_obj.diff_dem
            diff_error_file = diff_dem_obj.diff_error
        files.append(diff_dem_file)
        diff_dem, count = mask_tif(diff_dem_file, polygon, all_touched=all_touched)
        error, _ = mask_tif(diff_error_file, polygon, all_touched=all_touched)
        if diff_dem is not None:
            diff_dems.append(diff_dem)
            errors.append(error)
            counts.append(count)

    if len(diff_dems) > 1:
        max_index =counts.index(max(counts))
        dh = diff_dems[max_index].flatten()
        sigma_dh = errors[max_index].flatten()
    elif len(diff_dems) == 1:
        dh = diff_dems[0].flatten()
        sigma_dh = errors[0].flatten()
    else:
        return None, None, None
    dh_new = dh[~np.isnan(dh)]
    sigma_dh = sigma_dh[~np.isnan(dh)]
    sigma_dh[np.isnan(sigma_dh)] = 0.0
    return dh_new, sigma_dh, files[0]

def mask_tif(file, polygon, all_touched=False):
    try:
        with rio.open(file) as opened_file:
            geom = mapping(polygon)
            out_img, _ = mask(opened_file, [geom], crop=True, nodata=-9999, all_touched=all_touched)
            new_file = np.where((out_img > 8000) | (out_img < -500), np.nan, out_img)
            count = np.count_nonzero(~np.isnan(new_file))
            opened_file.close()
    except:
        new_file = None
        count = None
    return new_file, count

def is_contained_in_mask(polygon, tif_file, zero_threshold):
    with rio.open(tif_file) as src:
        geom = mapping(polygon)
        out_img, _ = mask(src, [geom], crop=True)
        out_img = np.where((out_img > 8000) | (out_img < -500), np.nan, out_img)
        zero_count = (out_img == 0.0).sum()
        total_pixels = np.count_nonzero(~np.isnan(out_img))
        zero_percentage = zero_count / total_pixels
        mask_filter = True if zero_percentage >= zero_threshold else False
    return mask_filter