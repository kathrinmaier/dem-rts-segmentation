
def select_data(cfg, mode): # add logic to select data based on mode
    path = f"{cfg.paths.root}{cfg.paths.data}"
    if mode == "fit":
        
        data = {"train": [[f"{path}/train"], [f"{path}/train/mask.tif"]], 
                "val": [[f"{path}/val"], [f"{path}/val/mask.tif"]]
                }
    elif mode == "test":
        data = [[f"{path}/test"],[f"{path}/test/mask.tif"]]
    elif mode == "predict":
        data = [f"{path}/predict"]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return data

