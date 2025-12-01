"""
This script demonstrates how to train a model using the stable-SSL library.
"""
import sys
import os

# Add the parent folder to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))


import hydra
from omegaconf import DictConfig

from models.ssl_models.stable_ssl_patches import patch_stable_ssl
from models.ssl_models.custom_config import get_args

from models.ssl_models.custom_supervised import Supervised
from models.ssl_models.custom_barlow_twins import BarlowTwins
from models.ssl_models.factored_models import CovarianceFactorization, MaskingFactorization


model_dict = {
    "Supervised": Supervised,
    "BarlowTwins": BarlowTwins,
    "CovarianceFactorization": CovarianceFactorization,
    "MaskingFactorization": MaskingFactorization,
}


@hydra.main(config_path="configs/ssl_configs/")
def main(cfg: DictConfig):
    changed = patch_stable_ssl()
    print(f"Applied {len(changed)} patches to stable-ssl!")
    args = get_args(cfg)

    print("--- Arguments ---")
    print(args)

    trainer = model_dict[args.model.name](args)
    trainer()


if __name__ == "__main__":
    main()
