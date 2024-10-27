import logging
from typing import Dict, Any, Optional, Union, Mapping
import time
from datetime import datetime
import os 
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

from src.logger.base_logger import BaseLogger
from src.utils.logger import RankedLogger


class RankedConsoleLogger(BaseLogger):
    """支持多GPU的控制台日志记录器"""
    
    def __init__(
        self,
        exp_name: str = None,
        rank_zero_only: bool = True,
        log_level: int = logging.INFO
    ):
        self.exp_name = exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.step_times = []
        self._last_step_time = self.start_time
        
        # 设置ranked logger
        self.logger = RankedLogger(
            name=f"experiment_{self.exp_name}",
            rank_zero_only=rank_zero_only
        )
        self.logger.setLevel(log_level)
        
        # 设置控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.logger.addHandler(console_handler)
        
    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _format_metric(self, name: str, value: float) -> str:
        if isinstance(value, (float, np.float32, np.float64)):
            return f"{name}: {value:.4f}"
        return f"{name}: {value}"
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        current_time = time.time()
        elapsed = current_time - self.start_time
        step_time = current_time - self._last_step_time
        self.step_times.append(step_time)
        
        avg_step_time = np.mean(self.step_times[-100:])
        
        # 构建日志消息
        step_str = f"Step {step}" if step is not None else ""
        time_str = f"[{self._format_time(elapsed)}]"
        metrics_str = " | ".join(self._format_metric(k, v) for k, v in metrics.items())
        step_time_str = f"({step_time:.2f}s/step, avg: {avg_step_time:.2f}s)"
        
        message = f"{time_str} {step_str} {metrics_str} {step_time_str}"
        self.logger.log(logging.INFO, message)
        
        self._last_step_time = current_time
    
    def log_hyperparams(self, params: Dict[str, Any]):
        self.logger.log(logging.INFO, "\nHyperparameters:")
        for k, v in params.items():
            self.logger.log(logging.INFO, f"{k}: {v}")
        self.logger.log(logging.INFO, "")
    
    def finish(self):
        total_time = time.time() - self.start_time
        self.logger.log(
            logging.INFO,
            f"\nTraining finished in {self._format_time(total_time)}"
        )