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
    os.makedirs(f"{ROOT}/compare/stats/{process_dict['run_id']}", exist_ok=True)

    name = f"{process_dict['year_start']}_{process_dict['year_end']}"

    for mode in ["plain", "postprocessed"]:
        ref_gdf = gpd.read_file(f"{ROOT}/compare/reference/polygons_reference_{mode}.geojson", crs=f"EPSG:{crs}")
        comp_gdf = gpd.read_file(f"{ROOT}/predict/postprocess/{process_dict['run_id']}/polygons_prediction_{mode}.geojson", crs=f"EPSG:{crs}")

        ref_gdf["ref_id"] = range(1, len(ref_gdf) + 1)
        comp_gdf["comp_id"] = range(1, len(comp_gdf) + 1)
        ref_gdf = filter_by_year(ref_gdf, process_dict)
        comp_gdf = filter_by_year(comp_gdf, process_dict)
        
        df = mass_wasting.compute_stats(ref_gdf, comp_gdf)

        df.to_csv(f"{ROOT}/compare/stats/{process_dict['run_id']}/stats_segmentation.csv", index=False)
        
        df = mass_wasting.compare_mass_wasting_performance(ref_gdf, comp_gdf, (int(process_dict["year_end"]) - int(process_dict["year_start"])))
        if df is not None:
            df.to_csv(f"{ROOT}/compare/stats/{process_dict['run_id']}/stats_dem.csv", index=False)
    return

def filter_by_year(gdf, process_dict):
    return gdf[(gdf.year_start.astype(str) == str(process_dict['year_start'])) & 
                (gdf.year_end.astype(str) == str(process_dict['year_end']))]

if __name__ == "__main__":
    main()  