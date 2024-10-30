from typing import Callable, Dict, Sequence, Union, Tuple, Optional

import design_bench as db 
import numpy as np 
import jax.numpy as jnp 

from design_bench.task import Task

from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset

from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
from design_bench.datasets.discrete.nas_bench_dataset import NASBenchDataset

from vizier import pyvizier as vz 

from src.task.base_task import OfflineBBOExperimenter

_task_kwargs = {
    "AntMorphology-Exact-v0":{
        "relabel": True
    },
    "TFBind10-Exact-v0": {
        "dataset_kwargs": {
            "max_samples": 10000,
        }
    }
}

_taskname2datafunc = {
    "AntMorphology-Exact-v0": lambda: AntMorphologyDataset(),
    "DKittyMorphology-Exact-v0": lambda: DKittyMorphologyDataset(),
    "Superconductor-RandomForest-v0": lambda: SuperconductorDataset(),
    "TFBind8-Exact-v0": lambda: TFBind8Dataset(),
    "TFBind10-Exact-v0": lambda: TFBind10Dataset(),
    "CIFARNAS-Exact-v0": lambda: NASBenchDataset(),
}

def load_ood_data(
    task_name: str, 
    task_x: np.ndarray,
    task_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    full_dataset = _taskname2datafunc[task_name]()
    full_x: np.ndarray = full_dataset.x.copy() 
    full_y: np.ndarray = full_dataset.y.copy()
    
    def create_mask(full_x: np.ndarray, x: np.ndarray) -> np.ndarray:
        
        dtype = [('', full_x.dtype)] * full_x.shape[1]
        
        full_x_struct = full_x.view(dtype)
        task_x_struct = x.view(dtype)

        mask = np.in1d(full_x_struct, task_x_struct)
        return mask
    
    if task_name == "TFBind10-Exact-v0":
        index = np.random.choice(full_y.shape[0], 30000, replace=False)
        full_x = full_x[index]
        full_y = full_y[index]
    
    mask = create_mask(full_x, task_x)
    diff_x = full_x[~mask] 
    diff_x, unique_indices = np.unique(diff_x, axis=0, return_index=True) 
    
    diff_y = full_y[~mask][unique_indices]
    
    indices = np.arange(diff_x.shape[0]) 
    np.random.shuffle(indices) 
    diff_x = diff_x[indices] 
    diff_y = diff_y[indices]

    return diff_x, diff_y


class DesignBenchExperimenter(OfflineBBOExperimenter):
    
    def __init__(
        self, 
        task_name: str,
        require_to_logits: bool = True,
        require_normalize_xs: bool = True,
        require_normalize_ys: bool = True,
    ) -> None:
        self.task: Task = db.make(task_name, **_task_kwargs.get(task_name, {}))
        dic2y = np.load("dic2y.npy", allow_pickle=True).item() 
        full_y_min, full_y_max = dic2y[task_name]
        
        task_x, task_y = self.task.x.copy(), self.task.y.copy()
        x_ood, y_ood = load_ood_data(task_name, task_x, task_y)
        
        super(DesignBenchExperimenter, self).__init__(
            task_name=task_name,
            eval_fn=self.task.predict,
            x_np=task_x,
            y_np=task_y,
            full_y_min=full_y_min,
            full_y_max=full_y_max,
            is_discrete=self.task.is_discrete,
            x_ood_np=x_ood,
            y_ood_np=y_ood,
            require_to_logits=require_to_logits,
            require_normalize_xs=require_normalize_xs,
            require_normalize_ys=require_normalize_ys,
        )
        
    @property
    def num_classes(self) -> Optional[int]:
        if self.task.is_discrete:
            return self.task.num_classes
        return None
    
    def problem_statement(self) -> vz.ProblemStatement:
        problem_statement = vz.ProblemStatement()
        root = problem_statement.search_space.root
        
        if self.task.is_discrete:
            for i in range(self.input_size):
                root.add_categorical_param(name=f"x{i}", feasible_values=list(range(self.task.num_classes)))
                
        else:
            for i in range(self.input_size):
                root.add_float_param(name=f"x{i}", min_value=float("-inf"), max_value=float("inf")) 
                
        problem_statement.metric_information.extend(
            [
                vz.MetricInformation(name="Score", goal=vz.ObjectiveMetricGoal.MAXIMIZE),
                vz.MetricInformation(name="Normalized_Score", goal=vz.ObjectiveMetricGoal.MAXIMIZE),
            ]
        )
        return problem_statement
    
    def normalize_x(self, x: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        self._shape0 = x.shape[1:]
        return self.task.normalize_x(x).reshape(x.shape[0], -1)
    
    def normalize_y(self, y: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        return self.task.normalize_y(y)
    
    def denormalize_x(self, x: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        return self.task.denormalize_x(x.reshape(x.shape[0], *self._shape0))
    
    def denormalize_y(self, y: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        return self.task.denormalize_y(y)
    
    def to_logits(self, x: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        return self.task.to_logits(x)
    
    def to_integers(self, x: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        return self.task.to_integers(x)
