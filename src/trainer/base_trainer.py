from copy import deepcopy
from functools import partial
import abc 
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
from src._typing import PRNGKeyArray as KeyArray

log = RankedLogger(__name__, rank_zero_only=True)

class Trainer(abc.ABC):
    def __init__(
        self,
        data_module: JAXDataModule,
        metrics: Optional[Dict[str, Metric]] = None,
        seed: int = 0,
        rng: KeyArray = jax.random.PRNGKey(0),
        save_prefix: str = "",
        checkpoint_dir: Union[str, os.PathLike] = './checkpoints',
        logger: Optional[Union[BaseLogger, list[BaseLogger]]] = None,
    ):
        self.data_module = data_module
        self.seed = seed
        self.rng = rng

        self.save_prefix = f"{save_prefix}-" if save_prefix else ""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        
        if logger is None:
            self.logger = WandBLogger()
        elif isinstance(logger, list):
            self.logger = MultiLogger(logger)
        else:
            self.logger = logger
        self.metrics = metrics or {"mse": MSE()}
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': []
        }
        for metric_name in self.metrics:
            self.history[f'train_{metric_name}'] = []
            self.history[f'val_{metric_name}'] = []


    def _reset_metrics(self):
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset()

    def _update_metrics(self, preds: jnp.ndarray, targets: jnp.ndarray):
        """Update all metrics"""
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def on_fit_start(self):
        tmp_history = deepcopy(self.history)
        for key in self.history.keys():
            if key == "epoch_times":
                continue
            tmp_history[f"best_{key}"] = []
        self.history = tmp_history
        
    def on_fit_end(self):
        pass

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self, train_loss: float, train_metrics: Dict[str, float]):
        if self.history["best_train_loss"] == []:
            self.history["best_train_loss"].append(train_loss)
        else:
            self.history["best_train_loss"].append(
                min(self.history["best_train_loss"][-1], train_loss)
            )
        
        for name, metric in train_metrics.items():
            if self.history[f"best_train_{name}"] == []:
                self.history[f"best_train_{name}"].append(metric)
            else:
                self.history[f"best_train_{name}"].append(
                    max(self.history[f"best_train_{name}"][-1], metric) \
                        if self.metrics[name].maximize else \
                            min(self.history[f"best_train_{name}"][-1], metric)
                )

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self, val_loss: float, val_metrics: Dict[str, float]):
        if self.history["best_val_loss"] == []:
            self.history["best_val_loss"].append(val_loss)
        else:
            self.history["best_val_loss"].append(
                min(self.history["best_val_loss"][-1], val_loss)
            )
        
        for name, metric in val_metrics.items():
            if self.history[f"best_val_{name}"] == []:
                self.history[f"best_val_{name}"].append(metric)
            else:
                self.history[f"best_val_{name}"].append(
                    max(self.history[f"best_val_{name}"][-1], metric) \
                        if self.metrics[name].maximize else \
                            min(self.history[f"best_val_{name}"][-1], metric)
                )

    def on_test_epoch_start(self):
        pass

    def on_test_epoch_end(self, test_loss: float, test_metrics: Dict[str, float]):
        if self.history["best_test_loss"] == []:
            self.history["best_test_loss"].append(test_loss)
        else:
            self.history["best_test_loss"].append(
                min(self.history["best_test_loss"][-1], test_loss)
            )
        
        for name, metric in test_metrics.items():
            if self.history[f"best_test_{name}"] == []:
                self.history[f"best_test_{name}"].append(metric)
            else:
                self.history[f"best_test_{name}"].append(
                    max(self.history[f"best_test_{name}"][-1], metric) \
                        if self.metrics[name].maximize else \
                            min(self.history[f"best_test_{name}"][-1], metric)
                )

    def save_checkpoint(self, params, checkpoint_name: str):
        logging.getLogger().setLevel(logging.WARNING)
        save_path = self.checkpoint_dir / checkpoint_name
        self.checkpointer.save(
            save_path, 
            params,
            force=True  
        )
        logging.getLogger().setLevel(logging.INFO)
        
    def load_checkpoint(self, checkpoint_name: str):
        load_path = self.checkpoint_dir / checkpoint_name
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {load_path}")
        return self.checkpointer.restore(load_path)

    def create_train_state(
        self, 
        rng: KeyArray, 
        input_shape: Tuple,
        dtype: Union[Type[jnp.int32], Type[jnp.int64], Type[jnp.float32], Type[jnp.float64]] = jnp.float32,
    ) -> train_state.TrainState:
        """Initial model params and optimizer state"""
        if dtype not in [jnp.int32, jnp.int64, jnp.float32, jnp.float64]:
            raise ValueError("dtype must be either jnp.int, jnp.float, or jnp.double")
            
        params = self.model.init(rng, jnp.ones(input_shape, dtype=dtype))
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
        )

    @abc.abstractmethod
    def train_step(self, state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[train_state.TrainState, float, float]:
        """Single train step"""
        pass

    @abc.abstractmethod
    def eval_step(self, state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[train_state.TrainState, float, float]:
        """Single eval step"""
        pass

    def train_epoch(self, state: train_state.TrainState, rng: KeyArray)\
        -> Tuple[train_state.TrainState, float, Dict[str, float]]:
        """Train one epoch"""
        self.on_train_epoch_start()
        self._reset_metrics()
        
        x_batches, y_batches = self.data_module.train_dataloader(rng)
        batch_losses = []
        
        for x_batch, y_batch, w_batch in zip(x_batches, y_batches):
            state, loss, preds = self.train_step(state, (x_batch, y_batch))
            batch_losses.append(loss)
            self._update_metrics(preds, y_batch)
            self.global_step += 1
            
        epoch_loss = jnp.mean(jnp.array(batch_losses))
        metrics = self._compute_metrics()
        
        self.on_train_epoch_end(epoch_loss, metrics)
            
        return state, epoch_loss, metrics

    def validate_epoch(self, state: train_state.TrainState, rng: KeyArray)\
        -> Tuple[float, Dict[str, float]]:
        """Validate one epoch"""
        self.on_validation_epoch_start()
        self._reset_metrics()
        
        x_batches, y_batches = self.data_module.val_dataloader(rng)
        batch_losses = []
        
        for x_batch, y_batch in zip(x_batches, y_batches):
            state, loss, preds = self.eval_step(state, (x_batch, y_batch))
            batch_losses.append(loss)
            self._update_metrics(preds, y_batch)
            self.global_step += 1
            
        epoch_loss = jnp.mean(jnp.array(batch_losses))
        metrics = self._compute_metrics()
        
        self.on_validation_epoch_end(epoch_loss, metrics)
            
        return epoch_loss, metrics

    def test(self, state: train_state.TrainState, rng: KeyArray)\
        -> Tuple[float, Dict[str, float]]:
        """Test the model"""
        self.on_test_epoch_start()
        self._reset_metrics()
        
        x_batches, y_batches = self.data_module.test_dataloader(rng)
        batch_losses = []
        all_preds = []
        
        for x_batch, y_batch in zip(x_batches, y_batches):
            state, loss, preds = self.eval_step(state, (x_batch, y_batch))
            batch_losses.append(loss)
            self._update_metrics(preds, y_batch)
            all_preds.append(preds)
            
        epoch_loss = jnp.mean(jnp.array(batch_losses))
        metrics = self._compute_metrics()
        
        self.on_test_epoch_end(epoch_loss, metrics)
            
        return epoch_loss, metrics

    def fit(self, model: nn.Module, input_shape: Tuple[int]):
        """Fit the model"""
        self.model = model
        self.on_fit_start()
        
        self.rng, init_rng, train_rng = jax.random.split(self.rng, 3)
        epoch_rng = jax.random.split(train_rng, self.max_epochs)
        self.state = self.create_train_state(init_rng, input_shape, self.data_module.input_dtype)
        
        for epoch in tqdm(range(self.max_epochs), desc="Training"):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # train one epoch
            self.rng, epoch_rng = jax.random.split(self.rng)
            train_rng, val_rng = jax.random.split(epoch_rng)
            self.state, train_loss, train_metrics = self.train_epoch(self.state, train_rng)
            
            # validate
            val_loss, val_metrics = self.validate_epoch(self.state, val_rng)
            
            # record time
            epoch_time = time.time() - epoch_start_time
            
            # update history
            self.history['train_loss'].append(float(train_loss))
            self.history['val_loss'].append(float(val_loss))
            self.history['epoch_times'].append(epoch_time)
            
            for metric_name, value in train_metrics.items():
                self.history[f'train_{metric_name}'].append(value)
            for metric_name, value in val_metrics.items():
                self.history[f'val_{metric_name}'].append(value)
            
            # save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.save_best_val_epoch:
                    self.best_params = self.state.params.copy()
                    self.save_checkpoint(self.best_params, f'{self.save_prefix}best_model.ckpt')
            
            if (epoch + 1) % self.save_checkpoint_epochs == 0:
                self.save_checkpoint(self.state.params, f'{self.save_prefix}epoch_{epoch+1}.ckpt')
            
            # log
            metrics_to_log = {k: v[-1] for k, v in self.history.items() if v != []}
            self.logger.log_metrics(metrics_to_log, step=self.global_step)
            
            # print(f"\nEpoch {epoch+1}/{self.max_epochs}")
            # print(f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            # print(f"time: {epoch_time:.2f}s")
            
        self.save_checkpoint(self.state.params, f'{self.save_prefix}last_model.ckpt')
        
        if self.eval_test:
            self.rng, test_rng = jax.random.split(self.rng)
            if self.save_best_val_epoch and self.best_params is not None:
                test_state = self.state.replace(params=self.best_params)
            else:
                test_state = self.state
                
            test_loss, test_metrics = self.test(test_state, test_rng)
            
            self.history['test_loss'].append(float(test_loss))
            for metric_name, value in test_metrics.items():
                self.history[f'test_{metric_name}'].append(value)
            
            test_results = {
                'test_loss': float(test_loss),
                **{f'test_{k}': v for k, v in test_metrics.items()}
            }
            self.logger.log_metrics(test_results)
            log.info("\nTest Results:")
            log.info(f"test_loss: {test_loss:.4f}")
            for metric_name, value in test_metrics.items():
                log.info(f"test_{metric_name}: {value:.4f}")
        
        self.on_fit_end()

    def predict(self, x: jnp.ndarray, params: Optional[Any] = None) -> jnp.ndarray:
        """Predict using the model"""
        if params is None:
            if self.save_best_val_epoch and self.best_params is not None:
                params = self.best_params
            else:
                params = self.state.params
        return self.state.apply_fn(params, x)
    
    def save_history(self) -> None:
        """save history"""
        history_path = self.checkpoint_dir / f'{self.save_prefix}training_history.json'
        history_dict = {k: [float(v) for v in vals] for k, vals in self.history.items()}
        history_dict['best_epoch'] = self.best_epoch
        history_dict['best_val_loss'] = float(self.best_val_loss)
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
            
    def load_best_model(self):
        """Load the best model"""
        self.state = self.state.replace(
            params=self.load_checkpoint(f'{self.save_prefix}best_model.ckpt')
        )
        return self.state
    
    def load_last_model(self):
        """Load the final model"""
        self.state = self.state.replace(
            params=self.load_checkpoint(f'{self.save_prefix}last_model.ckpt')
        )
        return self.state
    
    def load_model(self):
        if self.save_best_val_epoch:
            self.state = self.load_best_model()
        else:
            self.state = self.load_last_model()
        return self.state

    def get_history(self) -> Dict[str, float]:
        """Return history"""
        return self.history