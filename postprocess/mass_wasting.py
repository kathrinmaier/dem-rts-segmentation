import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import postprocess.utils as util

def calculate_area(dh, pixel_area=100):
    area_total = len(dh[dh != 0.0]) * pixel_area
    area_negative = len(dh[(dh < 0.0)]) * pixel_area
    area_positive = len(dh[(dh > 0.0)]) * pixel_area
    return area_total, area_negative, area_positive


def calculate_height(dh, sigma_dh):
    dh_net = np.sum(dh) 
    sigma_dh_net =  np.sqrt(np.sum(sigma_dh**2)) 
    dh_loss = np.sum(dh[dh < 0.0]) 
    sigma_dh_loss =  np.sqrt(np.sum(sigma_dh[dh < 0.0]**2))
    return dh_loss, sigma_dh_loss, dh_net, sigma_dh_net

def calculate_height_bounds(dh, sigma_dh, dh_big=None, sigma_dh_big=None):
    if dh_big is not None and sigma_dh_big is not None:
        dh_upper = np.array([height - err for height, err in zip(dh_big, sigma_dh_big)])   
    else:
        dh_upper = np.array([height - err for height, err in zip(dh, sigma_dh)])    
    dh_lower = np.array([height + err for height, err in zip(dh, sigma_dh)])
    return dh_upper, dh_lower

def calculate_volume(dh, sigma_dh, pixel_area=100):
    dv_net = np.sum(dh * pixel_area)
    sigma_dv_net = np.sqrt(np.nansum((sigma_dh * pixel_area) **2))
    dv_loss = np.sum(dh[dh < 0.0] * pixel_area)
    sigma_dv_loss = np.sqrt(np.nansum((sigma_dh[dh < 0.0] * pixel_area) **2))
    return dv_loss, sigma_dv_loss, dv_net, sigma_dv_net

def calculate_volume_bounds(dh_upper, dh_lower, pixel_area=100):
    dv_upper = np.sum(dh_upper[dh_upper < 0.0] * pixel_area)
    dv_lower = np.sum(dh_lower[dh_lower < 0.0] * pixel_area)
    return dv_upper, dv_lower


def calculate_volume_in_depth(dh, stats, pixel_area=100):
    depths = [(0.0, -1.0), (-1.0, -2.0), (-2.0, -3.0), (-3.0, None), (-3.0, -5.0), (-5.0, -10.0), (-10.0, None)]
    for (upper, lower) in depths:
        complete_fit = np.full_like(dh, -1)
        if lower is not None:
            dh_depth = dh[(dh > lower) & (dh < upper)]
            depth_layer = np.full_like(dh_depth, 1) * upper
            partial_fit = dh_depth - depth_layer
            tot_dh_depth = np.nansum(partial_fit) + np.nansum(complete_fit[dh <= lower])
            stats[f"dv_{abs(upper):.0f}_{abs(lower):.0f}m"] = tot_dh_depth * pixel_area

        else:
            dh_depth = dh[dh < upper]
            depth_layer = np.full_like(dh_depth, 1) * upper
            partial_fit = dh_depth - depth_layer
            tot_dh_depth = np.nansum(partial_fit)
            stats[f"dv_under{abs(upper):.0f}m"] = tot_dh_depth * pixel_area
    return stats

def add_attributes(diff_dem_objects, polygon, time_span):
    stats = {}
    stats["time_span"] = time_span
    dh, sigma_dh, _ = util.read_and_write_dem_data(diff_dem_objects, polygon, all_touched=False)
    stats["area"], stats["area_neg"], stats["area_pos"] = calculate_area(dh)
    stats["dh_loss"], stats["sigma_dh_loss"], stats["dh_net"], stats["sigma_dh_net"] = calculate_height(dh, sigma_dh)
    dh_upper, dh_lower = calculate_height_bounds(dh, sigma_dh)
    if dh is None:
        return None
    stats["dv_loss"], stats["sigma_dv_loss"], stats["dv_net"], stats["sigma_dv_net"] = calculate_volume(dh, sigma_dh)
    stats["dv_upper"], stats["dv_lower"] = calculate_volume_bounds(dh_upper, dh_lower)
    stats = calculate_volume_in_depth(dh, stats)
    stats = postprocess_stats(stats)
    return stats

def postprocess_polygon(polygon, diff_dem_list):
    remove = False
    water_contained = any(util.is_contained_in_mask(polygon, diff_dem["watermask"], 0.2) for diff_dem in diff_dem_list)
    sar_contained = any(util.is_contained_in_mask(polygon, diff_dem["sar_mask"], 0.5) for diff_dem in diff_dem_list)
    if water_contained or sar_contained or polygon.area <= 1000:
        remove = True
    return remove

def calculate_polygon_attributes(process_dict, gdf, temp_path, data_path):
    tmp_txt_file_dict = {}
    for temp_txt_file in ['soc_mishra', 'soc_wang', 'alt', 'gi']:
        tmp_txt_file_dict[temp_txt_file] = f"{temp_path}/attribute_{process_dict['year_start']}_{process_dict['year_end']}_{temp_txt_file}.txt"

    crs = int(process_dict['crs'])
    time_span = int(process_dict["year_end"]) - int(process_dict["year_start"])
    polygons_to_remove = []
    polygons_stats = {}
    total = gdf.shape[0]
    for idx, polygon in gdf.geometry.items():
        print(f"{idx} of {total}")
        updated_diff_dem_list = [] 
        updated_watermask_list = []
        diff_dem_dict = {}
        for filename in os.listdir(data_path):
            if filename.endswith(".tif"):
                pattern = pattern = r'^([a-z_]+)'
                key_temp = re.match(pattern, filename).group(1)[:-1] # type: ignore
                diff_dem_dict[key_temp] = f"{data_path}/{filename}"
        updated_diff_dem_list.append(diff_dem_dict)
        
        stats = add_attributes(updated_diff_dem_list, polygon, tmp_txt_file_dict, process_dict, time_span=time_span, crs=crs)  

        if stats is None or polygon is None:
            print("Problem with height calculation.")
            continue
        else:
            polygons_stats[idx] = stats

        gdf.at[idx, 'geometry'] = polygon

        remove = postprocess_polygon(polygon, updated_diff_dem_list, updated_watermask_list, process_dict)
        if remove:
            print(f"Polygon {idx} is removed.")
            polygons_to_remove.append(idx)
            
    df = pd.DataFrame.from_dict(polygons_stats, orient='index')
    gdf = gdf.merge(df, left_index=True, right_index=True)
    if "area_y" in gdf.columns:
        gdf.rename(columns={"area_y": "area"}, inplace=True)
        gdf.drop(columns=["area_x"], inplace=True)

    for txt_file in tmp_txt_file_dict.values():
        if os.path.exists(txt_file):
            os.remove(txt_file)
    return gdf, polygons_to_remove



def compare_mass_wasting_performance(gdf_ref, gdf_comp, time_span=10):
    dem_dict = {}
    for key, gdf in {"ref": gdf_ref, "comp": gdf_comp}.items():
        key_dict = {}
        key_dict["type"] = key
        key_dict["timespan"] = time_span
        key_dict["num_polygons"] = len(gdf)
        
        # Area, Volume, and Height Change
        sum_list = ["area", "area_neg", "area_pos", "dh_loss",  "dh_net", "dv_loss", "dv_upper", "dv_lower", "dv_net", "dv_0_1m", "dv_1_2m", "dv_2_3m", "dv_under3m", "dv_3_5m", "dv_5_10m", "dv_under10m"]
        error_list = ["sigma_dh_loss", "sigma_dv_loss", "sigma_dv_net"]
        for attribute in sum_list:
            key_dict[attribute] = np.nansum(gdf[attribute])
        for attribute in error_list:
            key_dict[attribute] = np.sqrt(np.nansum(gdf[attribute]**2))

        dem_dict[key] = key_dict

    ratio_dict = {}
    ratio_dict["type"] = "ratio"
    for key in dem_dict["comp"]:
        if key not in ["type", "timespan", "num_polygons"]:
            ratio_dict[key] = (dem_dict["comp"][key] - dem_dict["ref"][key]) / dem_dict["ref"][key]
    
    dem_dict["ratio"] = ratio_dict
    return pd.DataFrame.from_dict(dem_dict, orient='index')

def compute_stats(ref_gdf, comp_gdf):
    intersection = gpd.overlay(comp_gdf, ref_gdf, how='intersection', keep_geom_type=False)
    union = gpd.overlay(comp_gdf, ref_gdf, how='union', keep_geom_type=False)
    metrics = {
        "num_ref": len(ref_gdf),
        "num_comp": len(comp_gdf),
        "tp": intersection['ref_id'].nunique(),
        "fn": len(ref_gdf) - intersection['ref_id'].nunique(),
        "fp": len(comp_gdf) - intersection['ref_id'].nunique(),
        "pixel_iou": intersection.geometry.area.sum() / union.geometry.area.sum(),
        "pixel_p": intersection.geometry.area.sum() / comp_gdf.geometry.unary_union.area,
        "pixel_r": intersection.geometry.area.sum() / ref_gdf.geometry.area.sum(),
        "detection_iou_mean": np.mean([
            row1.geometry.intersection(row2.geometry).area / row1.geometry.union(row2.geometry).area
            for _, row1 in ref_gdf.iterrows()
            for _, row2 in comp_gdf.iterrows()
            if row1.geometry.intersects(row2.geometry)
        ])
    }
    metrics["detection_iou"] = metrics["tp"] / metrics["tp"] + metrics["fp"] + metrics["fn"] 
    metrics["pixel_f1"] = 2 * (metrics["pixel_p"] * metrics["pixel_r"]) / (metrics["pixel_p"] + metrics["pixel_r"])
    metrics["detection_p"] = metrics["tp"] / (metrics["tp"] + metrics["fp"])
    metrics["detection_r"] = metrics["tp"] / (metrics["tp"] + metrics["fn"])
    metrics["detection_f1"] = 2 * (metrics["detection_p"] * metrics["detection_r"]) / (metrics["detection_p"] + metrics["detection_r"])
    return pd.DataFrame([metrics])

def postprocess_stats(stats):
    for key in stats.keys():
        if isinstance(stats[key], (float, np.float64, np.float32)): # type: ignore
            stats[key] = round(stats[key], 2)
        if "dh" in key or "dv" in key:
            stats[key] = abs(stats[key])
        if "lower" in key:
            if stats[key] < 0.0:
                stats[key] = 0.0
    return stats

