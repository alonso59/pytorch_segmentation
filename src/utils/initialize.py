import os
import sys
import yaml
import torch
import logging
import datetime
import numpy as np


def initialize(cfg, bayessian=None):
    """Directories"""
    now = datetime.datetime.now()
    log_path = 'logs/' + cfg['model_name'] + "_"+ str(now.strftime("%m_%d_%H_%M_%S")) + '/' 
    checkpoint_path = log_path + "checkpoints/"

    create_dir(checkpoint_path)

    with open(log_path + 'experiment_cfg.yaml', 'w') as yaml_config:
        yaml.dump(cfg, yaml_config)

    """logging"""
    logging.basicConfig(filename=log_path + "info.log",
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    """ Seeding """
    # seeding(43)  # 42
    return logger, checkpoint_path, log_path

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seeding(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True