import os
import geopandas as gpd
import postprocess.utils as util
import postprocess.mass_wasting as mass_wasting

ROOT = "/data"
TEMP = "/temp"

def main():
    ini_location = 'postprocess/postprocess.ini'
    process_dict = util.get_config_dict(filename=ini_location,
                                            section='general')
    crs = int(process_dict['crs'])
    year_start = int(process_dict['year_start'])
    year_end = int(process_dict['year_end'])

    os.makedirs(f"{ROOT}/validation", exist_ok=True)

    name = f"{year_start}_{year_end}"
    polygon_gdf = gpd.read_file(f"{ROOT}/test/polygons/polygons_reference.geojson", crs=f"EPSG:{crs}")
    polygon_gdf, polygons_to_remove = mass_wasting.calculate_polygon_attributes(process_dict, polygon_gdf, TEMP, f"{ROOT}/test/images")

    polygon_gdf.to_file(f"{ROOT}/compare/validation/polygons_plain_{name}.geojson", driver="GeoJSON", crs=f"EPSG:{crs}")
    polygon_gdf = polygon_gdf.drop(polygons_to_remove)
    polygon_gdf.to_file(f"{ROOT}/compare/validation/polygons_postprocessed_{name}.geojson", driver="GeoJSON", crs=f"EPSG:{crs}")
    return

if __name__ == "__main__":
    main()  