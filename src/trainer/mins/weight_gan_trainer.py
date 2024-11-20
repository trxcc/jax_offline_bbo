from typing import Any, Callable, Tuple, List, Tuple, Dict, Optional, Union, Type

import flax.linen as nn 
from flax.training import train_state
from functools import partial
import jax 
import jax.numpy as jnp 

import optax 
import orbax.checkpoint 
import os 
import time 
from tqdm import tqdm 

from src.data.datamodule import JAXDataModule
from src.metric.base_metric import Metric
from src.logger.base_logger import BaseLogger
from src.utils.logger import RankedLogger
from src.trainer.base_trainer import Trainer
from src._typing import PRNGKeyArray as KeyArray
from src.trainer.mins.utils import cont_noise, disc_noise

log = RankedLogger(__name__, rank_zero_only=True)

class WeightGANTrainer(Trainer):
    def __init__(
        self, 
        data_module: JAXDataModule,
        critic_frequency: int,
        flip_frac: float,
        fake_pair_frac: float, 
        penalty_weight: float,
        generator_opt: optax.GradientTransformation,
        discriminator_opt: optax.GradientTransformation,
        is_discrete: bool,
        metrics: Optional[Dict[str, Metric]] = None,
        max_epochs: int = 100,
        seed: int = 0,
        rng: KeyArray = jax.random.PRNGKey(0),
        noise_std: float = 0.0,
        keep: float = 1.0,
        start_temp: float = 5.0,
        final_temp: float = 1.0,
        save_prefix: str = "",
        save_checkpoint_epochs: int = 10,
        checkpoint_dir: Union[str, os.PathLike] = './checkpoints',
        logger: Optional[Union[BaseLogger, list[BaseLogger]]] = None,
    ) -> None:
        super(WeightGANTrainer, self).__init__(
            data_module=data_module,
            metrics=metrics,
            seed=seed,
            rng=rng,
            save_prefix=save_prefix,
            checkpoint_dir=checkpoint_dir,
            logger=logger
        )
        self.is_discrete = is_discrete
        self.noise_std = noise_std
        self.keep = keep
        self.critic_frequency = critic_frequency
        self.penalty_weight = penalty_weight
        self.fake_pair_frac = fake_pair_frac
        self.flip_frac = flip_frac
        
        self.start_temp = start_temp
        self.final_temp = final_temp
        self.temp = jnp.array(0.0, dtype=jnp.float32)
        
        self.generator_opt = generator_opt
        self.discriminator_opt = discriminator_opt
        
        self.max_epochs = max_epochs
        self.save_checkpoint_epochs = save_checkpoint_epochs
        
        self.global_step = 0
    
    
    def create_train_state(
        self, 
        rng: KeyArray, 
        input_shape: Tuple[int],
        dtype: Union[Type[jnp.int32], Type[jnp.int64], Type[jnp.float32], Type[jnp.float64]] = jnp.float32,
    ) -> Dict[str, train_state.TrainState]:
        if dtype not in [jnp.int32, jnp.int64, jnp.float32, jnp.float64]:
            raise ValueError("dtype must be either jnp.int, jnp.float, or jnp.double")
        batch_size = input_shape[0]
        
        g_rng, d_rng = jax.random.split(rng)
        g_params = self.generator.init(
            g_rng, 
            jnp.ones((batch_size, 1), dtype=dtype),
            jnp.ones((batch_size, self.generator.latent_size), dtype=dtype)
        )
        g_state = train_state.TrainState.create(
            apply_fn=self.generator.apply,
            params=g_params,
            tx=self.generator_opt
        )
        
        # Initialize discriminator
        d_params = self.discriminator.init(
            d_rng, 
            jnp.ones(input_shape, dtype=dtype),
            jnp.ones((batch_size, 1), dtype=dtype)
        )
        d_state = train_state.TrainState.create(
            apply_fn=self.discriminator.apply,
            params=d_params,
            tx=self.discriminator_opt
        )
        
        return {"generator": g_state, "discriminator": d_state}
    
    @partial(jax.jit, static_argnames=['self']) 
    def compute_fake_loss(self, d_params, g_params, y_real, rng):
        rng, z_key = jax.random.split(rng)
        z = jax.random.normal(z_key, (y_real.shape[0], self.generator.latent_size))
        x_fake = self.generator.apply(
            g_params,
            y_real, z, temp=self.temp, train=False,
        )
        p_fake, d_fake, acc_fake = self.discriminator.loss(
            x_fake, y_real, 
            jnp.zeros((y_real.shape[0], 1)), 
            params=d_params
        )
        return x_fake, p_fake, d_fake, acc_fake, rng
    
    @partial(jax.jit, static_argnames=['self'])
    def compute_pair_loss(self, d_params, y_real, x_real, rng):
        rng, shuffle_key = jax.random.split(rng)
        x_pair = jax.random.shuffle(shuffle_key, x_real)

        p_pair, d_pair, acc_pair = self.discriminator.loss(
            x_pair, y_real,
            jnp.zeros((y_real.shape[0], 1)),
            params=d_params
        )
        return x_pair, p_pair, d_pair, acc_pair, rng
    
    @partial(jax.jit, static_argnames=['self'])
    def compute_real_loss(self, d_params, y_real, x_real, rng):
        rng, tmp_key = jax.random.split(rng)
        labels = (self.flip_frac <= \
            jax.random.uniform(tmp_key, shape=(y_real.shape[0], 1))).astype(jnp.float32)
        p_real, d_real, acc_real = self.discriminator.loss(
            x_real, y_real,
            labels, params=d_params 
        )
        return p_real, d_real, acc_real, rng
        
    @partial(jax.jit, static_argnames=['self'])
    def compute_penalty(self, x_fake, x_real, y_real, d_params, rng):
        rng, gp_key = jax.random.split(rng)
        e = jax.random.uniform(gp_key, shape=[y_real.shape[0]] + [1] * (len(x_fake.shape) - 1))
        x_interp = x_real * e + x_fake * (1 - e)
        penalty = self.discriminator.penalty(x_interp, y_real, params=d_params)
        return penalty, rng
    
    @partial(jax.jit, static_argnames=['self'])
    def compute_total_loss(self, w, d_real, d_fake, penalty):
        return jnp.mean(w * (d_real + d_fake + self.penalty_weight * penalty))
    
    
    def train_step(
        self, 
        state: Dict[str, train_state.TrainState], 
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        rng: KeyArray,
    ) -> Tuple[Dict[str, train_state.TrainState], Dict[str, float], jnp.ndarray]:
        """Training step for WeightGAN"""
        metrics = {} 
        
        x_real, y_real, w = batch
        batch_size = y_real.shape[0]
        
        rng, noise_key = jax.random.split(rng)
        if self.is_discrete:
            x_real = disc_noise(x_real, noise_key, keep=self.keep, temp=self.temp) 
        else:
            x_real = cont_noise(x_real, noise_key, self.noise_std)
        
        g_state = state["generator"]
        d_state = state["discriminator"]
        
        def discriminator_loss_fn(d_params, rng):
            (
                x_fake, p_fake, d_fake, acc_fake, rng
            ) = self.compute_fake_loss(
                d_params, g_state.params, y_real, rng
            )
            
            metrics['generator/train/y_real'] = jnp.mean(y_real)
            metrics['discriminator/train/p_fake'] = jnp.mean(p_fake)
            metrics['discriminator/train/d_fake'] = jnp.mean(d_fake)
            metrics['discriminator/train/acc_fake'] = jnp.mean(acc_fake)
            
            d_fake = d_fake * (1.0 - self.fake_pair_frac)
            
            p_pair = jnp.zeros_like(p_fake)
            d_pair = jnp.zeros_like(d_fake) 
            acc_pair = jnp.zeros_like(acc_fake)
            
            if self.fake_pair_frac > 0:
                
                (
                    x_pair, p_pair, d_pair, acc_pair, rng
                ) = self.compute_pair_loss(
                    d_params, y_real, x_real, rng
                )
                
                d_fake = d_pair * self.fake_pair_frac + d_fake 
            
            metrics['discriminator/train/p_pair'] = jnp.mean(p_pair)
            metrics['discriminator/train/d_pair'] = jnp.mean(d_pair)
            metrics['discriminator/train/acc_pair'] = jnp.mean(acc_pair)
            
            (
                p_real, d_real, acc_real, rng
            ) = self.compute_real_loss(
                d_params, y_real, x_real, rng
            )
            
            metrics['discriminator/train/p_real'] = jnp.mean(p_real)
            metrics['discriminator/train/d_real'] = jnp.mean(d_real)
            metrics['discriminator/train/acc_real'] = jnp.mean(acc_real)
            
            penalty, rng = self.compute_penalty(x_fake, x_real, y_real, d_params, rng)
            
            metrics['discriminator/train/neg_critic_loss'] = jnp.mean(-(d_real + d_fake))
            metrics['discriminator/train/penalty'] = jnp.mean(penalty)
            
            total_loss = self.compute_total_loss(w, d_real, d_fake, penalty)
            return total_loss
        
        total_loss, d_grads = jax.value_and_grad(
            lambda x: discriminator_loss_fn(x, rng=rng))(d_state.params)
        d_state = d_state.apply_gradients(grads=d_grads)
        metrics['discriminator/train/loss'] = total_loss
        
       # Update generator 
        def update_generator(rng):
            
            @jax.jit
            def g_loss_fn(g_params, loss_rng):
                z = jax.random.normal(loss_rng, (batch_size, self.generator.latent_size))
                x_fake = self.generator.apply(
                    g_params,
                    y_real, z, temp=self.temp, train=True
                )
                p_fake, d_fake, _ = self.discriminator.loss(
                    x_fake, y_real,
                    jnp.ones((batch_size, 1)),
                    params=d_state.params
                )
                loss = w * d_fake 
                return loss.mean()
             
            g_grads = jax.grad(lambda x: g_loss_fn(x, loss_rng=rng))(g_state.params)
            return g_state.apply_gradients(grads=g_grads)

        # Conditionally update generator based on iteration count
        g_state = jax.lax.cond(
            self.current_epoch % self.critic_frequency == 0,
            lambda: update_generator(rng=rng),
            lambda: g_state
        )
        
        state = {"generator": g_state, "discriminator": d_state}
        return state, metrics

    def eval_step(
        self, 
        state: Dict[str, train_state.TrainState], 
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        rng: KeyArray,
    ) -> Tuple[Dict[str, train_state.TrainState], Dict[str, float], jnp.ndarray]:
        """Evaluation step for WeightGAN"""
        metrics = {} 
        
        x_real, y_real = batch
        batch_size = y_real.shape[0]
        
        rng, noise_key = jax.random.split(rng)
        if self.is_discrete:
            x_real = disc_noise(x_real, noise_key, keep=self.keep, temp=self.temp) 
        else:
            x_real = cont_noise(x_real, noise_key, self.noise_std)
            
        g_state = state["generator"]
        d_state = state["discriminator"]
        
        x_fake, p_fake, d_fake, acc_fake, rng = self.compute_fake_loss(
            d_state.params, g_state.params, y_real, rng
        )
        
        metrics[f'generator/validate/y_real'] = jnp.mean(y_real)
        metrics[f'discriminator/validate/p_fake'] = jnp.mean(p_fake)
        metrics[f'discriminator/validate/d_fake'] = jnp.mean(d_fake)
        metrics[f'discriminator/validate/acc_fake'] = jnp.mean(acc_fake)
        
        p_pair = jnp.zeros_like(p_fake)
        d_pair = jnp.zeros_like(d_fake)
        acc_pair = jnp.zeros_like(acc_fake)
        
        if self.fake_pair_frac > 0:
            
            x_pair, p_pair, d_pair, acc_pair, rng = self.compute_pair_loss(
                d_state.params, y_real, x_real, rng
            )
            
        metrics['discriminator/validate/p_pair'] = jnp.mean(p_pair)
        metrics['discriminator/validate/d_pair'] = jnp.mean(d_pair)
        metrics['discriminator/validate/acc_pair'] = jnp.mean(acc_pair)
        
        @jax.jit
        def compute_d_real(x_real, y_real, d_params):
            p_real, d_real, acc_real = self.discriminator.loss(
                x_real, y_real, 
                jnp.ones((y_real.shape[0], 1)),
                params=d_params
            )
            return d_real
        
        d_real = compute_d_real(x_real, y_real, d_state.params)
        
        penalty, rng = self.compute_penalty(x_fake, x_real, y_real, d_state.params, rng)
        
        metrics['discriminator/validate/neg_critic_loss'] = jnp.mean(-(d_real + d_fake))
        metrics['discriminator/validate/penalty'] = jnp.mean(penalty)
        
        total_loss = jnp.mean(d_real + d_fake + self.penalty_weight * penalty)
        metrics['discriminator/validate/loss'] = total_loss
        return state, metrics
    
    def _update_metrics(self, metrics: Dict[str, float]):
        for name, metric in metrics.items():
            if isinstance(metric, jnp.ndarray):
                metric = jax.device_get(metric).item()
            if name not in self.history.keys():
                self.history[name] = [metric]
            else:
                self.history[name].append(metric)
    
    def on_train_epoch_end(self, train_loss: float):
        if self.history["best_train_loss"] == []:
            self.history["best_train_loss"].append(train_loss)
        else:
            self.history["best_train_loss"].append(
                min(self.history["best_train_loss"][-1], train_loss)
            )
            
    def on_validation_epoch_end(self, validate_loss: float):
        if self.history["best_val_loss"] == []:
            self.history["best_val_loss"].append(validate_loss)
        else:
            self.history["best_val_loss"].append(
                min(self.history["best_val_loss"][-1], validate_loss)
            )
    
    def train_epoch(self, state: train_state.TrainState, rng: KeyArray) \
        -> Tuple[Dict[str, train_state.TrainState], float, Dict[str, float]]:
        """Train one epoch"""
        self.on_train_epoch_start()
        # self._reset_metrics()
        rng, data_loader_key = jax.random.split(rng)
        x_batches, y_batches, w_batches = self.data_module.train_dataloader(data_loader_key)
        batch_losses = []
        
        for x_batch, y_batch, w_batch in zip(x_batches, y_batches, w_batches):
            rng, step_key = jax.random.split(rng)
            state, metrics = self.train_step(state, (x_batch, y_batch, w_batch), step_key)
            self._update_metrics(metrics)
            batch_losses.append(metrics['discriminator/train/loss'])
            self.global_step += 1
        
        epoch_loss = jnp.mean(jnp.array(batch_losses))
        self.on_train_epoch_end(epoch_loss)
            
        return state, epoch_loss
    
    def validate_epoch(self, state: train_state.TrainState, rng: KeyArray) \
        -> float:
        
        self.on_validation_epoch_start()
        # self._reset_metrics()
        rng, data_loader_key = jax.random.split(rng)
        x_batches, y_batches = self.data_module.val_dataloader(data_loader_key)
        batch_losses = []
        
        for x_batch, y_batch in zip(x_batches, y_batches):
            rng, step_key = jax.random.split(rng)
            state, metrics = self.eval_step(state, (x_batch, y_batch), step_key)
            self._update_metrics(metrics)
            batch_losses.append(metrics['discriminator/validate/loss'])
            self.global_step += 1
        
        epoch_loss = jnp.mean(jnp.array(batch_losses))
        self.on_validation_epoch_end(epoch_loss)
            
        return epoch_loss
    

    def fit(self, generator: nn.Module, discriminator: nn.Module, input_shape: Tuple[int]):
        self.generator = generator
        self.discriminator = discriminator
        
        self.on_fit_start() 
        
        self.rng, init_rng, train_rng = jax.random.split(self.rng, 3)
        
        self.state = self.create_train_state(init_rng, input_shape, self.data_module.input_dtype)
        
        
        # TODO: OOM
        for epoch in tqdm(range(self.max_epochs), desc="Training GAN"):
            self.current_epoch = epoch 
            epoch_start_time = time.time()
        
            self.rng, epoch_rng = jax.random.split(self.rng)
            train_rng, val_rng = jax.random.split(epoch_rng)
            
            self.temp = jnp.array(self.final_temp * epoch / (self.max_epochs - 1) +
                             self.start_temp * (1.0 - epoch / (self.max_epochs - 1)),
                             dtype=jnp.float32)
            
            self.state, train_loss = self.train_epoch(self.state, train_rng)
            val_loss = self.validate_epoch(self.state, val_rng)
            
            epoch_time = time.time() - epoch_start_time
            
            # Update history
            self.history['epoch_times'].append(epoch_time)
            
            if (epoch + 1) % self.save_checkpoint_epochs == 0:
                current_params = {
                    'generator': self.state['generator'].params,
                    'discriminator': self.state['discriminator'].params
                }
                self.save_checkpoint(current_params, f'{self.save_prefix}epoch_{epoch+1}.ckpt')
            
            # Log metrics
            metrics_to_log = {k: v[-1] for k, v in self.history.items() if v != []}
            self.logger.log_metrics(metrics_to_log, step=self.global_step)
        
        # Save final model
        final_params = {
            'generator': self.state['generator'].params,
            'discriminator': self.state['discriminator'].params
        }
        self.save_checkpoint(final_params, f'{self.save_prefix}last_model.ckpt')
        
        self.on_fit_end()
        
    def load_checkpoint(self, checkpoint_name: str):
        load_path = self.checkpoint_dir / checkpoint_name
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {load_path}")
        return self.checkpointer.restore(load_path)
            
    def load_model(self):
        tmp_state = self.load_checkpoint(f"{self.save_prefix}last_model.ckpt")
        self.generator = self.generator.replace(params=tmp_state["generator"].params)
        
        