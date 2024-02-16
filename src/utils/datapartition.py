import sys
import hydra
import logging
import torch
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from datahandler import DataSetHandler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import seed_everything

@hydra.main(config_path="../config", config_name="data_config")
def main(cfg:DictConfig):
    seed_val = 2024
    seed_everything(seed_val)
    log = logging.getLogger("DATA PARTITION")
    datahandler = DataSetHandler(cfg.data)
    path_to_save = Path(cfg.data.partition_dir)
    trainchunks, testdata = datahandler()
    log.info("Data Partitioning Complete")
    path_to_save = Path(cfg.data.partition_dir)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    for idx,train in enumerate(trainchunks):
        log.info(f"Client {idx} Train: {len(train)}")
        # save the data partitioning
        client_path = path_to_save/f"client_{idx}"
        if not os.path.exists(client_path):
            os.makedirs(client_path)
        torch.save(train, client_path/f"trainset_{idx}.pt")
    torch.save(testdata, path_to_save/f"testset.pt")

if __name__ == "__main__":
    main()