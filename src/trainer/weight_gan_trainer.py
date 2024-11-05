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
from src.trainer.utils.noise import cont_noise, disc_noise

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
    
    
    def create_train_state(
        self, 
        rng: KeyArray, 
        input_shape: Tuple,
        dtype: Union[Type[jnp.int32], Type[jnp.int64], Type[jnp.float32], Type[jnp.float64]] = jnp.float32,
    ) -> Dict[str, train_state.TrainState]:
        if dtype not in [jnp.int32, jnp.int64, jnp.float32, jnp.float64]:
            raise ValueError("dtype must be either jnp.int, jnp.float, or jnp.double")
        
        g_rng, d_rng = jax.random.split(rng)
        g_params = self.discriminator.init(g_rng, jnp.ones(input_shape, dtype=dtype))
        g_state = train_state.TrainState.create(
            apply_fn=self.generator.apply,
            params=g_params,
            tx=self.generator_opt
        )
        
        # Initialize discriminator
        d_params = self.discriminator.init(d_rng, jnp.ones(input_shape, dtype=dtype))
        d_state = train_state.TrainState.create(
            apply_fn=self.discriminator.apply,
            params=d_params,
            tx=self.discriminator_opt
        )
        
        return {"generator": g_state, "discriminator": d_state}
    
    
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
        batch_dim = y_real.shape[1]
        
        rng, noise_key = jax.random.split(rng)
        if self.is_discrete:
            x_real = disc_noise(x_real, noise_key, keep=self.keep, temp=self.temp) 
        else:
            x_real = cont_noise(x_real, noise_key, self.noise_std)
        
        g_state = state["generator"]
        d_state = state["discriminator"]
        
        def discriminator_loss_fn(d_params):
            rng, z_key = jax.random.split(rng)
            z = jax.random.normal(z_key, (batch_size, self.generator.latent_size))
            x_fake = self.generator.apply(
                {'params': g_state.params}, 
                y_real, z, temp=self.temp, train=False,
            )
            p_fake, d_fake, acc_fake = self.discriminator.loss(
                x_fake, y_real, 
                jnp.zeros((batch_size, 1)), 
                params=d_params
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
                
                rng, shuffle_key = jax.random.split(rng)
                x_pair = jax.random.shuffle(shuffle_key, x_real)
            
                p_pair, d_pair, acc_pair = self.discriminator.loss(
                    x_pair, y_real,
                    jnp.ones((batch_dim, 1)),
                    params=d_params
                )
                
                d_fake = d_pair * self.fake_pair_frac + d_fake 
            
            metrics['discriminator/train/p_pair'] = jnp.mean(p_pair)
            metrics['discriminator/train/d_pair'] = jnp.mean(d_pair)
            metrics['discriminator/train/acc_pair'] = jnp.mean(acc_pair)
            
            rng, tmp_key = jax.random.split(rng)
            labels = (self.flip_frac <= \
                jax.random.uniform(tmp_key, shape=(batch_dim, 1))).astype(jnp.float32)
            p_real, d_real, acc_real = self.discriminator.loss(
                x_real, y_real,
                labels, params=d_params 
            )
            
            metrics['discriminator/train/p_real'] = jnp.mean(p_real)
            metrics['discriminator/train/d_real'] = jnp.mean(d_real)
            metrics['discriminator/train/acc_real'] = jnp.mean(acc_real)
            
            rng, gp_key = jax.random.split(rng)
            e = jax.random.uniform(gp_key, shape=[batch_dim] + [1] * (len(x_fake.shape) - 1))
            x_interp = x_real * e + x_fake * (1 - e)
            penalty = self.discriminator.penalty(x_interp, y_real, params=d_params)
            
            metrics['discriminator/train/neg_critic_loss'] = jnp.mean(-(d_real + d_fake))
            metrics['discriminator/train/penalty'] = jnp.mean(penalty)
            
            total_loss = jnp.mean(w * (d_real + d_fake + self.penalty_weight * penalty))
            return total_loss
        
        total_loss, d_grads = jax.value_and_grad(discriminator_loss_fn, has_aux=True)(d_state.params)
        d_state = d_state.apply_gradients(grads=d_grads)
        
        metrics['discriminator/train/loss'] = total_loss
        
       # Update generator 
        def update_generator():
            rng, gen_key = jax.random.split(rng)
            z = jax.random.normal(gen_key, (batch_size, self.generator.latent_size))
            
            def g_loss_fn(g_params):
                x_fake = self.generator.apply(
                    {'params': g_params}, 
                    y_real, z, temp=self.temp, train=True
                )
                pred, loss, _ = self.discriminator.loss(
                    x_fake, y_real,
                    jnp.ones((batch_size, 1)),
                    params=d_state.params
                )
                return loss.mean()
                
            g_loss, g_grads = jax.value_and_grad(g_loss_fn)(g_state.params)
            metrics['generator/train/loss'] = g_loss 
            return g_state.apply_gradients(grads=g_grads)

        # Conditionally update generator based on iteration count
        g_state = jax.lax.cond(
            self.global_step % self.n_critic == 0,
            lambda: update_generator(),
            lambda: g_state
        )
        
        state = {"generator": g_state, "discriminator": d_state}
        return state, metrics

    def eval_step(
        self, 
        state: Dict[str, train_state.TrainState], 
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[Dict[str, train_state.TrainState], Dict[str, float], jnp.ndarray]:
        """Evaluation step for WeightGAN"""
        metrics = {} 
        
        x_real, y_real, _ = batch
        batch_size = y_real.shape[0]
        batch_dim = y_real.shape[1]
        
        rng, noise_key = jax.random.split(rng)
        if self.is_discrete:
            x_real = disc_noise(x_real, noise_key, keep=self.keep, temp=self.temp) 
        else:
            x_real = cont_noise(x_real, noise_key, self.noise_std)
            
        g_state = state["generator"]
        d_state = state["discriminator"]
        
        rng, z_key = jax.random.split(rng)
        
        z = jax.random.normal(z_key, (batch_size, self.generator.latent_size))
        x_fake = self.generator.apply(
            {'params': g_state.params}, 
            y_real, z, temp=self.temp, train=False,
        )
        p_fake, d_fake, acc_fake = self.discriminator.loss(
            x_fake, y_real, 
            jnp.zeros((batch_size, 1)), 
            params=d_state.params
        )
        metrics[f'generator/validate/y_real'] = jnp.mean(y_real)
        metrics[f'discriminator/validate/p_fake'] =jnp.mean(p_fake)
        metrics[f'discriminator/validate/d_fake'] = jnp.mean(d_fake)
        metrics[f'discriminator/validate/acc_fake'] = jnp.mean(acc_fake)
        
        
        p_pair = jnp.zeros_like(p_fake)
        d_pair = jnp.zeros_like(d_fake)
        acc_pair = jnp.zeros_like(acc_fake)
        
        if self.fake_pair_frac > 0:
            
            rng, shuffle_key = jax.random.split(rng)
            x_pair = jax.random.shuffle(shuffle_key, x_real)
        
            p_pair, d_pair, acc_pair = self.discriminator.loss(
                x_pair, y_real,
                jnp.ones((batch_dim, 1)),
                params=d_state.params
            )
            
            d_fake = d_pair * self.fake_pair_frac + d_fake 
            
        metrics['discriminator/validate/p_pair'] = jnp.mean(p_pair)
        metrics['discriminator/validate/d_pair'] = jnp.mean(d_pair)
        metrics['discriminator/validate/acc_pair'] = jnp.mean(acc_pair)
        
        p_real, d_real, acc_real = self.discriminator.loss(
            x_real, y_real, 
            jnp.ones((batch_dim, 1)),
            params=d_state.params
        )
        
        rng, gp_key = jax.random.split(rng)
        e = jax.random.uniform(gp_key, shape=[batch_dim] + [1] * (len(x_fake.shape) - 1))
        x_interp = x_real * e + x_fake * (1 - e)
        penalty = self.discriminator.penalty(x_interp, y_real, params=d_state.params)
        
        metrics['discriminator/validate/neg_critic_loss'] = jnp.mean(-(d_real + d_fake))
        metrics['discriminator/validate/penalty'] = jnp.mean(penalty)
        
        return state, metrics

    def fit(self, generator: nn.Module, discriminator: nn.Module, input_shape: Tuple[int]):
        self.generator = generator
        self.discriminator = discriminator
        
        self.on_fit_start() 
        
        self.rng, init_rng, train_rng = jax.random.split(self.rng, 3)
        
        self.state = self.create_train_state(init_rng, input_shape, self.data_module.dtype)
        
        for epoch in tqdm(range(self.max_epochs), desc="Training GAN"):
            self.current_epoch = epoch 
            epoch_start_time = time.time()
        
            self.rng, epoch_rng = jax.random.split(self.rng)
            train_rng, val_rng = jax.random.split(epoch_rng)
            
            self.state, train_metrics, _ = self.train_epoch(self.state, train_rng)
            val_metrics, _ = self.validate_epoch(self.state, val_rng)
            
            epoch_time = time.time() - epoch_start_time
            
            # Update history
            self.history['epoch_times'].append(epoch_time)
            
            for metric_name, value in train_metrics.items():
                self.history[f'train_{metric_name}'].append(float(value))
            for metric_name, value in val_metrics.items():
                self.history[f'val_{metric_name}'].append(float(value))
            
            # Save models based on discriminator loss
            # val_loss = val_metrics.get('d_loss', float('inf'))
            # if val_loss < self.best_val_loss:
            #     self.best_val_loss = val_loss
            #     if self.save_best_val_epoch:
            #         self.best_params = {
            #             'generator': self.state['generator'].params.copy(),
            #             'discriminator': self.state['discriminator'].params.copy()
            #         }
            #         self.save_checkpoint(self.best_params, f'{self.save_prefix}best_model.ckpt')
            
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
            
            