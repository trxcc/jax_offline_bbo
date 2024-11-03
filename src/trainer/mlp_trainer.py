from copy import deepcopy
from functools import partial
import jax
import jax.numpy as jnp
import json
import logging
import optax
import os 
import orbax.checkpoint
from pathlib import Path
import flax.linen as nn 
from typing import Any, Callable, Dict, Optional, Tuple, Union, Type
from flax.training import train_state
import time
from tqdm.auto import tqdm

from src.data.datamodule import JAXDataModule
from src.metric.base_metric import Metric
from src.metric.mse import MSE
from src.logger.base_logger import BaseLogger, MultiLogger
from src.logger.wandb_logger import WandBLogger
from src.utils.logger import RankedLogger
from src.trainer.base_trainer import Trainer
from src._typing import PRNGKeyArray as KeyArray

log = RankedLogger(__name__, rank_zero_only=True)

class MLPTrainer(Trainer):
    def __init__(
        self,
        # model: nn.Module,
        loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        optimizer: optax.GradientTransformation,
        data_module: JAXDataModule,
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
        super(MLPTrainer, self).__init__(
            data_module=data_module,
            metrics=metrics,
            seed=seed,
            rng=rng,
            save_prefix=save_prefix,
            checkpoint_dir=checkpoint_dir,
            logger=logger
        )
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.eval_test = eval_test
        self.save_best_val_epoch = save_best_val_epoch
        self.save_checkpoint_epochs = save_checkpoint_epochs
        
        self.logger.log_hyperparams({
            'max_epochs': max_epochs,
            'seed': seed,
            'eval_test': eval_test,
            'save_best_val_epoch': save_best_val_epoch,
        })
        
        self.state = None
        self.best_val_loss = float('inf')
        self.best_params = None
        self.best_epoch = -1
        
        if eval_test:
            self.history['test_loss'] = []
            for metric_name in metrics:
                self.history[f'test_{metric_name}'] = []
        
        self.current_epoch = 0
        self.global_step = 0
    

    @partial(jax.jit, static_argnames=['self']) 
    def train_step(self, state: train_state.TrainState, batch: Tuple) -> Tuple[train_state.TrainState, float, float]:
        """Single train step"""
        x, y = batch
        
        def loss_fn(params):
            preds = state.apply_fn(params, x)
            loss = self.loss_fn(preds, y)
            return jnp.mean(loss), preds

        (loss, preds), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, loss, preds

    @partial(jax.jit, static_argnames=['self']) 
    def eval_step(self, state: train_state.TrainState, batch: Tuple) -> Tuple[train_state.TrainState, float, float]:
        """Single eval step"""
        x, y = batch
        preds = state.apply_fn(state.params, x)
        loss = jnp.mean(self.loss_fn(preds, y))
        return state, loss, preds


class DualMLPTrainer(MLPTrainer):
    
    def __init__(
        self,
        loss_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        *args, 
        **kwargs,
    ) -> None:
        return super(DualMLPTrainer, self).__init__(loss_fn=loss_fn, *args, **kwargs)
    
    @partial(jax.jit, static_argnames=['self'])
    def train_step(self, state: train_state.TrainState, batch: Tuple) -> Tuple[train_state.TrainState, float]:
        x, y = batch 
        
        def loss_fn(params):
            mean, std = state.apply_fn(params, x)
            loss = self.loss_fn(mean, std, y)
            return jnp.mean(loss), mean 
        
        (loss, mean), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, loss, mean 
    
    @partial(jax.jit, static_argnames=['self'])
    def eval_step(self, state: train_state.TrainState, batch: Tuple) -> float:
        x, y = batch 
        mean, std = state.apply_fn(state.params, x)
        loss = jnp.mean(self.loss_fn(mean, std, y))
        return state, loss, mean 
    
    
    def predict(self, x: jnp.ndarray, params: Optional[Any] = None) -> jnp.ndarray:
        """Predict using the model"""
        if params is None:
            if self.save_best_val_epoch and self.best_params is not None:
                params = self.best_params
            else:
                params = self.state.params
        mean, _ = self.state.apply_fn(params, x)
        return mean 