from typing import Callable, Dict, Sequence, Union, Tuple, Optional

import abc
import design_bench as db 
import numpy as np 
import jax.numpy as jnp 

from design_bench.task import Task
from vizier import pyvizier as vz 
from vizier.benchmarks import Experimenter

class OfflineBBOExperimenter(Experimenter):
    
    def __init__(
        self, 
        task_name: str,
        eval_fn: Callable[[Union[np.ndarray, jnp.ndarray]],
            Union[np.ndarray, jnp.ndarray]],
        x_np: np.ndarray,
        y_np: np.ndarray,
        full_y_min: Union[float, np.ndarray],
        full_y_max: Union[float, np.ndarray],
        is_discrete: bool,
        x_ood_np: Optional[np.ndarray] = None,
        y_ood_np: Optional[np.ndarray] = None,
        require_to_logits: bool = True,
        require_normalize_xs: bool = True,
        require_normalize_ys: bool = True,
    ) -> None:
        super().__init__()
        self.task_name = task_name 
        self.eval_fn = eval_fn
        
        self.x_np = x_np
        self.y_np = y_np 
        
        self.x_ood_np = x_ood_np
        self.y_ood_np = y_ood_np
        
        self.full_y_min = full_y_min
        self.full_y_max = full_y_max
        
        self.is_discrete = is_discrete
        self.require_to_logits = require_to_logits
        self.require_normalize_xs = require_normalize_xs
        self.require_normalize_ys = require_normalize_ys
    
    @property
    def x(self) -> np.ndarray:
        return self.x_np
    
    @property
    def y(self) -> np.ndarray:
        return self.y_np
    
    @property
    def x_ood(self) -> Optional[np.ndarray]:
        return self.x_ood_np
    
    @property
    def y_ood(self) -> Optional[np.ndarray]:
        return self.y_ood_np
    
    @property
    def dataset_size(self) -> int:
        return self.x_np.shape[0]
    
    @property
    def input_shape(self) -> Tuple[int]:
        return tuple(self.x_np.shape[1:])
    
    @property
    def input_size(self) -> int:
        return np.prod(self.x_np.shape[1:]).item()
    
    @property
    def num_classes(self) -> Optional[int]:
        return None
    
    @abc.abstractmethod
    def normalize_x(
        self, 
        x: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        pass 
    
    @abc.abstractmethod
    def normalize_y(
        self, 
        y: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        pass 
    
    @abc.abstractmethod
    def denormalize_x(
        self, 
        x: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        pass 
    
    @abc.abstractmethod
    def denormalize_y(
        self, 
        y: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        pass 
    
    @abc.abstractmethod
    def to_logits(
        self,
        x: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        pass 
    
    @abc.abstractmethod
    def to_integers(
        self,
        x: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        pass 
        
    def trial2array(self, suggestions: Sequence[vz.Trial]) -> jnp.ndarray:
        x_all = []
        for suggestion in suggestions:
            x = []
            for i in range(self.input_size):
                x.append(suggestion.parameters[f"x{i}"].value)
            x_all.append(x)
        return jnp.array(x_all)
    
    def array2trial(self, x_batch: Union[np.ndarray, jnp.ndarray]) -> Sequence[vz.Trial]:
        all_trials = []
        for x_i in x_batch:
            trial = vz.Trial()
            for j in range(self.input_size):
                trial.parameters[f"x{j}"] = x_i[j].item()
            all_trials.append(trial)
        return all_trials
    
    def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
        x_batch = self.trial2array(suggestions)
        self.score = self.eval_fn(x_batch)
        self.normalized_score = (self.score - self.full_y_min) / (self.full_y_max - self.full_y_min)
        for suggestion, score_i, nml_score_i in zip(suggestions, self.score, self.normalized_score):
            measurement = vz.Measurement(metrics={"Score": score_i, "Normalized_Score": nml_score_i})
            suggestion.complete(measurement)
    
    @abc.abstractmethod
    def problem_statement(self) -> vz.ProblemStatement:
        pass 
    
    def score(self, x_batch: Union[np.ndarray, jnp.ndarray]) -> Dict[str, np.ndarray]:
        trials = self.array2trial(x_batch)
        self.evaluate(trials)
        score = [] 
        normalized_score = [] 
        
        for trial in trials:
            score.append(trial.final_measurement.metrics["Score"].value)
            normalized_score.append(trial.final_measurement.metrics["Normalized_Score"].value)
        
        score = np.array(score)
        normalized_score = np.array(normalized_score)
        
        return {
            "Score": score,
            "Normalized_Score": normalized_score,
        }
        
    
    # def read_score_from_trials(self, trials: Sequence[vz.Trial]) -> Dict[str, float]:
    #     score = [] 
    #     normalized_score = [] 
        
    #     for trial in trials:
    #         score.append(trial.final_measurement.metrics["Score"].value)
    #         normalized_score.append(trial.final_measurement.metrics["Normalized_Score"].value)
        
    #     score = np.array(score)
    #     normalized_score = np.array(normalized_score)
        
    #     return {
    #         f"Score-25th": np.percentile(score, 25),
    #         f"Score-50th": np.percentile(score, 50),
    #         f"Score-75th": np.percentile(score, 75),
    #         f"Score-100th": np.percentile(score, 100),
    #         f"Normalized_Score-25th": np.percentile(normalized_score, 25),
    #         f"Normalized_Score-50th": np.percentile(normalized_score, 50),
    #         f"Normalized_Score-75th": np.percentile(normalized_score, 75),
    #         f"Normalized_Score-100th": np.percentile(normalized_score, 100),
    #     }
            