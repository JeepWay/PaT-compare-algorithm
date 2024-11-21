import numpy as np
import time
import csv
import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import ResultsWriter

class MetricsCallback(BaseCallback):
    """
    Custom callback for recording metrics in training process.    
    """
    
    EXT = "metrics.csv"
    
    def __init__(self, filename: str, verbose=0):
        super().__init__(verbose)
        if not filename.endswith(MetricsCallback.EXT):
            if os.path.isdir(filename):
                filename = os.path.join(filename, MetricsCallback.EXT)
            else:
                filename = filename + "." + MetricsCallback.EXT
        filename = os.path.realpath(filename)
        mode = "w"
        self.file_handler = open(filename, f"{mode}t", newline="\n")
        self.metrics_logger = csv.DictWriter(self.file_handler, fieldnames=(
            "timesteps",
            "ep_rew_mean",
            "ep_PE_mean",
            "loss",
            "policy_gradient_loss", 
            "value_loss", 
            "mask_loss", 
            "entropy_loss",
            "invalid_probs_loss",
            "learning_rate",)
        )
        self.metrics_logger.writeheader()
        self.file_handler.flush()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        metrics = {
            "timesteps": self.model.num_timesteps,
            "ep_rew_mean": self.logger.name_to_value['rollout/ep_rew_mean'], 
            "ep_PE_mean": self.logger.name_to_value['rollout/ep_PE_mean'], 
            "loss": self.logger.name_to_value['train/loss'], 
            "policy_gradient_loss": self.logger.name_to_value['train/policy_gradient_loss'], 
            "value_loss": self.logger.name_to_value['train/value_loss'], 
            "mask_loss": self.logger.name_to_value['train/mask_loss'], 
            "entropy_loss": self.logger.name_to_value['train/entropy_loss'], 
            "invalid_probs_loss": self.logger.name_to_value['train/invalid_probs_loss'], 
            "learning_rate": self.logger.name_to_value['train/learning_rate'], 
        }
        self.metrics_logger.writerow(metrics)
        self.file_handler.flush()

    def _on_training_end(self) -> None:
        self.file_handler.close()