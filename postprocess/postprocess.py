import os
import numpy as np
import geopandas as gpd
from shapely import box
import rasterio
import postprocess.utils as util
import postprocess.mass_wasting as mass_wasting


ROOT = "/data"
TEMP = "/temp"
DATA = "/data/model_output"

def main():
    ini_location = 'postprocess/postprocess.ini'
    process_dict = util.get_config_dict(filename=ini_location,
                                            section='general')
    crs = int(process_dict['crs'])
    os.makedirs(f"{ROOT}/postprocessed/{process_dict['run_id']}", exist_ok=True)
    name = f"{process_dict['year_start']}_{process_dict['year_end']}"               
    polygon_gdf, tile_gdf = preprare_predictions(f"{ROOT}/{process_dict['run_id']}/{name}", process_dict)
    polygon_gdf, polygons_to_remove = mass_wasting.calculate_polygon_attributes(process_dict, polygon_gdf, TEMP, DATA)
    polygon_gdf.to_file(f"{ROOT}/postprocessed/{process_dict['run_id']}/polygons_plain_{name}.geojson", driver="GeoJSON", crs=f"EPSG:{crs}")
    polygon_gdf = polygon_gdf.drop(polygons_to_remove)
    polygon_gdf.to_file(f"{ROOT}/postprocessed/{process_dict['run_id']}/polygons_postprocessed_{name}.geojson", driver="GeoJSON", crs=f"EPSG:{crs}")
    tile_gdf.to_file(f"{ROOT}/postprocessed/{process_dict['run_id']}/tiles_{name}.geojson", driver="GeoJSON", crs=f"EPSG:{crs}")
    return

def preprare_predictions(predict_path, process_dict):
    tiles = []
    polygons = []
    for file in [f"{predict_path}/{file}" for file in os.listdir(predict_path) if file.endswith(".tif")]:
        with rasterio.open(file) as src:
            bbox = src.bounds
            image, meta = src.read(1), src.meta
        # Tiles
        box_shape = box(bbox.left, bbox.bottom, bbox.right, bbox.top)
        tiles.append({"geometry": box_shape})
        # Polygons
        binary_image = np.zeros_like(image, dtype=np.uint8)
        binary_image[image > 0.5] = 1
        polygons.extend(util.polygonize(binary_image, meta['transform']))
    
    tiles_gdf = gpd.GeoDataFrame(geometry=tiles, crs=meta['crs']) # type: ignore
    polygon_gdf = gpd.GeoDataFrame(geometry=polygons, crs=meta['crs']) # type: ignore
    polygon_gdf = gpd.GeoDataFrame(
        geometry=[polygon_gdf.unary_union]).explode( # type: ignore
        index_parts=False).reset_index(
        drop=True)

    polygon_gdf["year_start"] = int(process_dict['year_start'])
    polygon_gdf["year_end"] = int(process_dict['year_end'])
    return polygon_gdf, tiles_gdf
                
  
if __name__ == "__main__":
    main()  