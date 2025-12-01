# policy_learning/train.py
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

@hydra.main(config_path="./configs/policy_learning", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # build the data generator (env-specific)
    data_gen = instantiate(cfg.env.dataset)

    # log training configs
    print("Env name:", cfg.env.environment_name)
    print("Total steps:", cfg.train.total_steps)
    print("PPO lr:", cfg.ppo.lr)

if __name__ == "__main__":
    main()
