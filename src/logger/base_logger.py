import abc 
from typing import Dict, Any, Optional

class BaseLogger(abc.ABC):
    
    @abc.abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass
    
    @abc.abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]):
        pass
    
    @abc.abstractmethod
    def finish(self):
        pass
    
class MultiLogger(BaseLogger):
    
    def __init__(self, loggers: list[BaseLogger]):
        self.loggers = loggers
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)
    
    def log_hyperparams(self, params: Dict[str, Any]):
        for logger in self.loggers:
            logger.log_hyperparams(params)
    
    def finish(self):
        for logger in self.loggers:
            logger.finish()