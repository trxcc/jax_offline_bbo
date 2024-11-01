from typing import Any, Callable, Dict, Optional, Union, Type, List, Tuple

from functools import partial
from copy import deepcopy
import flax.linen as nn 
import jax 
import jax.numpy as jnp 
import optax 
import os 

from src.data.datamodule import JAXDataModule
from src.metric.base_metric import Metric
from src.metric.mse import MSE 
from src.logger.base_logger import BaseLogger, MultiLogger
from src.logger.wandb_logger import WandBLogger
from src._typing import PRNGKeyArray as KeyArray
from src.trainer.base_trainer import Trainer
from src.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class EnsembleTrainer(Trainer):
    # TODO: Load and save checkpoint from specified path
    def __init__(
        self,
        # model: nn.Module,
        num_ensemble: int,
        base_trainer: Type[Trainer],
        loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        optimizer: optax.GradientTransformation,
        data_module: JAXDataModule,
        ensemble_type: str = "mean",
        metrics: Optional[Dict[str, Metric]] = None,
        max_epochs: int = 100,
        seed: int = 0,
        rng: KeyArray = jax.random.PRNGKey(0),
        eval_test: bool = True,
        save_best_val_epoch: bool = True,
        save_checkpoint_epochs: int = 10,
        save_prefix: str = "",
        checkpoint_dir: Union[str, os.PathLike] = './checkpoints',
        logger: Optional[Union[BaseLogger, list[BaseLogger]]] = None,
    ):
        self.ensemble_type = ensemble_type
        self.num_ensemble = num_ensemble
        self.key = rng
        self.save_prefix = f"{save_prefix}-" if save_prefix else ""
        
        self.trainers: List[Trainer] = []
        for i in range(num_ensemble):
            self.key, trainer_key = jax.random.split(self.key)
            trainer = base_trainer(
                loss_fn=loss_fn,
                optimizer=optimizer,
                data_module=data_module,
                metrics=deepcopy(metrics),
                max_epochs=max_epochs,
                seed=seed,
                rng=trainer_key,
                eval_test=eval_test,
                save_best_val_epoch=save_best_val_epoch,
                save_checkpoint_epochs=save_checkpoint_epochs,
                save_prefix=f"{self.save_prefix}model_{i}",
                checkpoint_dir=checkpoint_dir,
                logger=deepcopy(logger)
            )
            trainer.logger.set_prefix(f"model_{i}")
            self.trainers.append(trainer)
            
    def fit(self, model: Union[List[nn.Module], nn.Module], input_shape: Tuple[int]):
        if not isinstance(model, list):
            model = [deepcopy(model) for _ in range(self.num_ensemble)]
        elif len(model) != self.num_ensemble:
            model = [deepcopy(model[0]) for _ in range(self.num_ensemble)]
        
        for i, (model_i, trainer) in enumerate(zip(model, self.trainers)):
            log.info(f"Start training for model {i}")
            trainer.fit(model_i, input_shape)
    
    def predict(self, x: jnp.ndarray, params: Optional[List[Any]] = None) -> jnp.ndarray:
        """Predict using the model"""
        if params is None or len(params) != self.num_ensemble:
            params = [None for _ in range(self.num_ensemble)]
        
        predictions = []
        for param, trainer in zip(params, self.trainers):
            if param is None:
                if trainer.save_best_val_epoch and trainer.best_params is not None:
                    param = trainer.best_params
                else:
                    param = trainer.state.params
            predictions.append(trainer.predict(x, param))
        
        predictions = jnp.array(predictions)
        if self.ensemble_type == "mean":
            return predictions.mean(axis=0)
        elif self.ensemble_type == "min":
            return predictions.min(axis=0)
        else:
            raise NotImplementedError
    
    def load_best_model(self) -> None:
        params = []
        for trainer in self.trainers:
            params.append(trainer.load_best_model())
        return params 
    
    def load_last_model(self) -> None:
        params = []
        for trainer in self.trainers:
            params.append(trainer.load_last_model())
        return params
            
    def load_model(self) -> None:
        params = []
        for trainer in self.trainers:
            params.append(trainer.load_model())
        return params
    
    def get_history(self) -> Dict[str, float]:
        histories_list = [trainer.get_history() for trainer in self.trainers]
        histories = {} 
        for i, history in enumerate(histories_list):
            for k, v in history.items():
                histories[f"model_{i}-{k}"] = v     
        return histories
            
            