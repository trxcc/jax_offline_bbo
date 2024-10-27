from typing import Optional, Dict, Any

import wandb 
from datetime import datetime

from src.logger.base_logger import BaseLogger


class WandBLogger(BaseLogger):
    """Weights & Biases 日志记录器"""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        **kwargs
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags
        self.kwargs = kwargs
        
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            **kwargs
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if step is not None:
            metrics['step'] = step
        wandb.log(metrics)
    
    def log_hyperparams(self, params: Dict[str, Any]):
        wandb.config.update(params)
    
    def finish(self):
        wandb.finish()