from typing import Optional, Tuple, Dict, Any, Callable

import jax 
import jax.numpy as jnp
import random 
import torch 
import tensorflow as tf 
import numpy as np 
from omegaconf import DictConfig
from importlib.util import find_spec

from src._typing import PRNGKeyArray as KeyArray
from src.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

def seed_everything(seed: int) -> KeyArray:
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    return rng

def train_val_split(
    x: jnp.ndarray,
    y: jnp.ndarray,
    val_size: float = 0.2, 
    w: Optional[jnp.ndarray] = None,
    key: Optional[KeyArray] = None,
) -> Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]]: 
    
    if key is not None:
        key = jax.random.PRNGKey(0)
    
    idx = jax.random.permutation(key, len(x))
    split_idx = int(len(x) * (1 - val_size))
    
    train_idx, val_idx = idx[:split_idx], idx[split_idx:]
    
    if w is not None:
        return x[train_idx], y[train_idx], x[val_idx], y[val_idx], w[train_idx]
    else:
        return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict, score_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict, score_dict

    return wrap

def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

def obtain_percentile_score(
        score: np.ndarray,
        score_discription: str,
    ) -> Dict[str, float]:
        assert len(score.shape) == 1 or \
            (len(score.shape) == 2 and score.shape[1] == 1)
        score = score.flatten()
        
        return {
            f"{score_discription}-25th": np.percentile(score, 25),
            f"{score_discription}-50th": np.percentile(score, 50),
            f"{score_discription}-75th": np.percentile(score, 75),
            f"{score_discription}-100th": np.percentile(score, 100),
        }
