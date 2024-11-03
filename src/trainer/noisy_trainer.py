from typing import Any, Callable, Dict, Optional, Union, Type, List, Tuple

from functools import partial
from copy import deepcopy

import flax.linen as nn 
from flax.training.train_state import TrainState
import jax 
import jax.numpy as jnp 
import optax 
import os 
from jax.random import gumbel, normal

from src.data.datamodule import JAXDataModule
from src.metric.base_metric import Metric
from src.metric.mse import MSE 
from src.logger.base_logger import BaseLogger, MultiLogger
from src.logger.wandb_logger import WandBLogger
from src._typing import PRNGKeyArray as KeyArray
from src.trainer.mlp_trainer import Trainer
from src.trainer.ensemble_trainer import EnsembleTrainer
from src.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

@jax.jit
def disc_noise(
    x: jnp.ndarray, 
    key: KeyArray, 
    keep: float = 0.9, 
    temp: float = 5.0
) -> jnp.ndarray:
    p = jnp.ones_like(x)
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    p = keep * x + (1.0 - keep) * p
    gumbel_noise = gumbel(key, p.shape)
    return jax.nn.softmax((jnp.log(p) + gumbel_noise) / temp, axis=-1)

@jax.jit 
def cont_noise(
    x: jnp.ndarray, 
    key: KeyArray, 
    noise_std: float
) -> jnp.ndarray:
    return x + noise_std * normal(key, x.shape)

def override_train_epoch(original_method):
    def wrapper(self, state: TrainState, rng: KeyArray):
        self.on_train_epoch_start()
        self._reset_metrics()
        
        x_batches, y_batches = self.data_module.train_dataloader(rng)
        batch_losses = []
        
        for x_batch, y_batch in zip(x_batches, y_batches):
            x_batch = self.noise_fn(x_batch)
            state, loss, preds = self.train_step(state, (x_batch, y_batch))
            batch_losses.append(loss)
            self._update_metrics(preds, y_batch)
            self.global_step += 1
            
        epoch_loss = jnp.mean(jnp.array(batch_losses))
        metrics = self._compute_metrics()
        
        self.on_train_epoch_end(epoch_loss, metrics)
            
        return state, epoch_loss, metrics
    return wrapper

def override_validate_epoch(original_method):
    def wrapper(self, state: TrainState, rng: KeyArray):
        self.on_validation_epoch_start()
        self._reset_metrics()
        
        x_batches, y_batches = self.data_module.val_dataloader(rng)
        batch_losses = []
        
        for x_batch, y_batch in zip(x_batches, y_batches):
            x_batch = self.noise_fn(x_batch)
            state, loss, preds = self.eval_step(state, (x_batch, y_batch))
            batch_losses.append(loss)
            self._update_metrics(preds, y_batch)
            self.global_step += 1
            
        epoch_loss = jnp.mean(jnp.array(batch_losses))
        metrics = self._compute_metrics()
        
        self.on_validation_epoch_end(epoch_loss, metrics)
            
        return epoch_loss, metrics
    return wrapper

def override_test(original_method):
    def wrapper(self, state: TrainState, rng: KeyArray):
        self.on_test_epoch_start()
        self._reset_metrics()
        
        x_batches, y_batches = self.data_module.test_dataloader(rng)
        batch_losses = []
        all_preds = []
        
        for x_batch, y_batch in zip(x_batches, y_batches):
            x_batch = self.noise_fn(x_batch)
            state, loss, preds = self.eval_step(state, (x_batch, y_batch))
            batch_losses.append(loss)
            self._update_metrics(preds, y_batch)
            all_preds.append(preds)
            
        epoch_loss = jnp.mean(jnp.array(batch_losses))
        metrics = self._compute_metrics()
        
        self.on_test_epoch_end(epoch_loss, metrics)
            
        return epoch_loss, metrics
    return wrapper


class NoisyEnsembleTrainer(EnsembleTrainer):
    
    def __init__(
        self,
        num_ensemble: int,
        base_trainer: Type[Trainer],
        loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        optimizer: optax.GradientTransformation,
        data_module: JAXDataModule,
        keep: float = 0.0,
        temp: float = 0.0,
        noise_std: float = 0.0,
        is_discrete: bool = False,
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
    ) -> None:
        
        super(NoisyEnsembleTrainer, self).__init__(
            num_ensemble=num_ensemble,
            base_trainer=base_trainer,
            loss_fn=loss_fn,
            optimizer=optimizer,
            data_module=data_module,
            ensemble_type=ensemble_type,
            metrics=metrics,
            max_epochs=max_epochs,
            seed=seed,
            rng=rng,
            eval_test=eval_test,
            save_best_val_epoch=save_best_val_epoch,
            save_checkpoint_epochs=save_checkpoint_epochs,
            save_prefix=save_prefix,
            checkpoint_dir=checkpoint_dir,
            logger=logger
        )
        
        self.keep = keep 
        self.temp = temp 
        self.noise_std = noise_std
        self.is_discrete = is_discrete
        for trainer in self.trainers:
            trainer.rng, noise_key = jax.random.split(trainer.rng)
            trainer.noise_fn = lambda x: disc_noise(x, key=noise_key, keep=keep, temp=temp) \
                if is_discrete else lambda x: cont_noise(x, noise_std=noise_std)
            original_method = trainer.train_epoch
            trainer.train_epoch = override_train_epoch(original_method).__get__(trainer, type(trainer))

            original_method = trainer.validate_epoch
            trainer.validate_epoch = override_validate_epoch(original_method).__get__(trainer, type(trainer))
            
            original_method = trainer.test
            trainer.test = override_test(original_method).__get__(trainer, type(trainer))
        

            
        
    
        