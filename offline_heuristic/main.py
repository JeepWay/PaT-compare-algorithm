from typing import Dict, Any
import gymnasium as gym
import argparse
import yaml
import os
import shutil
import numpy as np
from collections import OrderedDict
from pprint import pprint
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='gymnasium.envs.registration')
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_system_info

from envs.register import registration_envs
from skyline_fit import SkylineBottomLeft


def test(config: Dict[str, Any]):
    print(f"\n{'-' * 30}   Start Testing   {'-' * 30}\n")
    ep_rewards_list = []
    ep_SUs_list = []

    for i in range(config["n_eval_seeds"]):
        eval_seed = int(config["eval_seed"] + i*10)
        env = gym.make(config['env_id'], **config["env_kwargs"]) 
        obs_shape = env.observation_space['bin'].shape
        ep_reward = 0
        
        if config.get("skyline_fit_kwargs") is not None:
            model = SkylineBottomLeft(
                bin_w=obs_shape[1],
                bin_h=obs_shape[2],
            )
        else:
            raise NotImplementedError("This heuristic method is not implemented yet.")

        for j in range(config['n_eval_episodes']):
            obs, info = env.reset(seed=eval_seed if j == 0 else None)
            logging.debug(f"episode 1, items_list: {env.unwrapped.items_creator.items_list}")
            
            items = env.unwrapped.items_creator.items_list
            items_sorted = sorted(items, key=lambda x: x[0] * x[1], reverse=True)
            env.unwrapped.items_creator.items_list = items_sorted
            obs = env.unwrapped._get_obs() 
            
            while True:
                action = model.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    ep_SUs_list.append(info["SU"])
                    ep_rewards_list.append(ep_reward)
                    ''' For skyline fit '''
                    model.reset()  
                    break
                  
    mean_reward = np.mean(ep_rewards_list)
    std_reward = np.std(ep_rewards_list)
    mean_SU = np.mean(ep_SUs_list)
    std_SU = np.std(ep_SUs_list)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"mean_SU: {mean_SU:.3f} +/- {std_SU:.3f}")
    with open(os.path.join(config['test_dir'], f"eval_{config['n_eval_episodes']}_{config['n_eval_seeds']}.txt"), "w") as file:
        file.write(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
        file.write(f"mean_SU: {mean_SU:.3f} +/- {std_SU:.3f}\n")
    print(f"\n{'-' * 30}   Complete Testing   {'-' * 30}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D BPP with heuristic methods")
    parser.add_argument('--config_path', default="settings/v1_skyline_bottom_left.yaml", type=str, help="Path to the configuration file with .yaml extension.")
    parser.add_argument('--mode', default="test", type=str, choices=["test"], help="Mode to test the model. Currently only 'test' is supported.")
    args = parser.parse_args()
    if not args.config_path.endswith(".yaml"):
        raise ValueError("Please specify the path to the configuration file with a .yaml extension.")

    # read hyperparameters from the .yaml config file
    with open(args.config_path, "r") as file:
        logging.info(f"Loading hyperparameters from: {args.config_path}")
        config = yaml.load(file, Loader=yaml.UnsafeLoader)

    # set `save_path` according to the name of the .yaml file
    config['save_path'] = os.path.join(config["log_dir"], 
        f"{config['env_id']}_{args.config_path.split('/')[-1][len('v1_'):-len('.yaml')]}"                 
    )

    # set test_dir according to the mode
    config['test_dir'] = config['save_path']
    os.makedirs(config['test_dir'], exist_ok=True)

    import shutil
    shutil.copy(args.config_path, os.path.join(config['test_dir'], args.config_path.split('/')[-1])) 

    # register custom environments
    registration_envs()

    print("\nSystem information: ")
    get_system_info(print_info=True)

    if args.mode == "test":
        test(config)
    else:
        raise ValueError("Invalid mode, please select either 'train' or 'test' or 'both'")

