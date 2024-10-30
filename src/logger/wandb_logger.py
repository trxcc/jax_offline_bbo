from typing import Optional, Dict, Any

import wandb 
from datetime import datetime

from src.logger.base_logger import BaseLogger


class WandBLogger(BaseLogger):
    """Weights & Biases Logger"""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        log_prefix: str = "",
        **kwargs
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags
        self.log_prefix = log_prefix
        self.kwargs = kwargs
        
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            **kwargs
        )
    
    def set_prefix(self, prefix: str = ""):
        self.log_prefix = prefix
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.log_prefix:
            metrics = {f"{self.log_prefix}/{k}": v for k, v in metrics.items()}
        if step is not None:
            metrics['step'] = step
        wandb.log(metrics)
    
    def log_hyperparams(self, params: Dict[str, Any]):
        if self.log_prefix:
            params = {f"{self.log_prefix}/{k}": v for k, v in params.items()}
        wandb.config.update(params)
    
    def finish(self):
        wandb.finish()