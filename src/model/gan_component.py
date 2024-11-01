import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple, Any 
import numpy as np
import distrax

class Discriminator(nn.Module):
    """A Fully Connected Network conditioned on a score"""
    
    design_shape: Sequence[int]
    hidden: int = 50
    method: str = 'wasserstein'

    def setup(self):
        """Initialize the model layers"""
        assert self.method in ['wasserstein', 'least_squares', 'binary_cross_entropy']
        
        self.input_size = np.prod(self.design_shape)
        
        # Define model layers
        self.embed_0 = nn.Dense(self.hidden)
        
        self.dense_0 = nn.Dense(self.hidden)
        self.ln_0 = nn.LayerNorm()
        
        self.dense_1 = nn.Dense(self.hidden)
        self.ln_1 = nn.LayerNorm()
        
        self.dense_2 = nn.Dense(self.hidden)
        self.ln_2 = nn.LayerNorm()
        
        self.dense_3 = nn.Dense(1)

    def __call__(self, x, y, train: bool = True):
        """Forward pass of the discriminator
        
        Args:
            x: input design (batch_size, *design_shape)
            y: scores (batch_size, 1)
            train: whether in training mode
        """
        x = x.astype(jnp.float32)
        y = y.astype(jnp.float32)
        
        # Reshape input
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Embed score
        y_embed = self.embed_0(y)
        
        # Forward pass through layers
        x = self.dense_0(jnp.concatenate([x, y_embed], axis=1))
        x = nn.leaky_relu(self.ln_0(x), negative_slope=0.2)
        x = self.dense_1(jnp.concatenate([x, y_embed], axis=1))
        x = nn.leaky_relu(self.ln_1(x), negative_slope=0.2)
        x = self.dense_2(jnp.concatenate([x, y_embed], axis=1))
        x = nn.leaky_relu(self.ln_2(x), negative_slope=0.2)
        return self.dense_3(jnp.concatenate([x, y_embed], axis=1))

    def penalty(self, h, y):
        """Calculate gradient penalty
        
        Args:
            h: input designs
            y: scores
        """
        # Calculate gradients
        grad_fn = jax.grad(lambda h: self.apply({'params': self.variables['params']}, h, y).sum())
        g = grad_fn(h)
        g = g.reshape(-1, self.input_size)
        return (1.0 - jnp.linalg.norm(g, axis=-1, keepdims=True)) ** 2

    def loss(self, x, y, labels):
        """Calculate discriminator loss
        
        Args:
            x: input designs
            y: scores  
            labels: binary labels indicating real/fake
        
        Returns:
            Tuple of (predictions, loss, accuracy)
        """
        pred = self.apply({'params': self.variables['params']}, x, y)
        
        if self.method == 'wasserstein':
            loss = jnp.where(labels > 0.5, -pred, pred)
            acc = jnp.where(labels > 0.5, 
                          (pred > 0.0).astype(jnp.float32),
                          (pred < 0.0).astype(jnp.float32))
            
        elif self.method == 'least_squares':
            loss = 0.5 * (pred - labels) ** 2
            acc = jnp.where(labels > 0.5,
                          (pred > 0.5).astype(jnp.float32),
                          (pred < 0.5).astype(jnp.float32))
            
        else:  # binary_cross_entropy
            pred = jax.nn.sigmoid(pred)
            loss = -labels * jnp.log(pred) - (1 - labels) * jnp.log(1 - pred)
            acc = jnp.where(labels > 0.5,
                          (pred > 0.5).astype(jnp.float32),
                          (pred < 0.5).astype(jnp.float32))
            
        return pred, loss, acc
    
    
class ConvDiscriminator(nn.Module):
    
    design_shape: Sequence[int]
    hidden: int = 50
    method: str = 'wasserstein'
    
    def setup(self):
        valid_methods = ['wasserstein', 'least_squares', 'binary_cross_entropy']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
            
        self.embed_0 = nn.Dense(self.hidden)
        
        self.conv_blocks = [
            (nn.Conv(self.hidden, kernel_size=(3,), strides=(2,), padding='SAME'),
             nn.LayerNorm()) for _ in range(3)
        ]
        
        self.final = nn.Dense(1)

    def __call__(self, x, y):
        x = x.astype(jnp.float32)
        y = y.astype(jnp.float32)
        
        y_embed = self.embed_0(y)
        
        def broadcast_embed(x, y_embed):
            return jnp.broadcast_to(
                y_embed[:, jnp.newaxis, :],
                (y_embed.shape[0], x.shape[1], y_embed.shape[1])
            )
        
        for conv, ln in self.conv_blocks:
            y_broadcast = broadcast_embed(x, y_embed)
            x = jnp.concatenate([x, y_broadcast], axis=2)
            x = conv(x)
            x = ln(x)
            x = jax.nn.leaky_relu(x, alpha=0.2)
        
        x = jnp.mean(x, axis=1)
        
        x = jnp.concatenate([x, y_embed], axis=1)
        return self.final(x)

    def penalty(self, h, y):
        grad_fn = jax.grad(lambda h: self.apply({'params': self.params}, h, y).sum())
        grads = grad_fn(h)
        grads = grads.reshape((grads.shape[0], -1))
        return (1.0 - jnp.linalg.norm(grads, axis=-1, keepdims=True)) ** 2

    def loss(self, x, y, labels):
        preds = self.apply({'params': self.params}, x, y)
        
        if self.method == 'wasserstein':
            loss = jnp.where(labels > 0.5, -preds, preds)
            acc = jnp.where(labels > 0.5, 
                          (preds > 0.0).astype(jnp.float32),
                          (preds < 0.0).astype(jnp.float32))
            
        elif self.method == 'least_squares':
            loss = 0.5 * (preds - labels) ** 2
            acc = jnp.where(labels > 0.5,
                          (preds > 0.5).astype(jnp.float32),
                          (preds < 0.5).astype(jnp.float32))
            
        else:  # binary_cross_entropy
            preds = jax.nn.sigmoid(preds)
            loss = -labels * jnp.log(preds) - (1 - labels) * jnp.log(1 - preds)
            acc = jnp.where(labels > 0.5,
                          (preds > 0.5).astype(jnp.float32),
                          (preds < 0.5).astype(jnp.float32))
            
        return preds, loss, acc
    


class DiscreteGenerator(nn.Module):
    
    design_shape: Sequence[int]
    latent_size: int
    hidden: int = 50
    
    def setup(self):
        self.embed_0 = nn.Dense(self.hidden)
        
        self.dense_0 = nn.Dense(self.hidden)
        self.ln_0 = nn.LayerNorm()
        
        self.dense_1 = nn.Dense(self.hidden)
        self.ln_1 = nn.LayerNorm()
        
        self.dense_2 = nn.Dense(self.hidden)
        self.ln_2 = nn.LayerNorm()
        
        self.dense_3 = nn.Dense(jnp.prod(jnp.array(self.design_shape)))
        
    def __call__(self, y, z, temp=1.0, train: bool = True):
        
        y = y.astype(jnp.float32)
        z = z.astype(jnp.float32)
        
        y_embed = self.embed_0(y)
        
        x = self.dense_0(jnp.concatenate([z, y_embed], axis=1))
        x = nn.leaky_relu(self.ln_0(x), negative_slope=0.2)
        x = self.dense_1(jnp.concatenate([x, y_embed], axis=1))
        
        x = nn.leaky_relu(self.ln_1(x), negative_slope=0.2)
        x = self.dense_2(jnp.concatenate([x, y_embed], axis=1))
        x = nn.leaky_relu(self.ln_2(x), negative_slope=0.2)
        x = self.dense_3(jnp.concatenate([x, y_embed], axis=1))
        
        batch_size = y.shape[0]
        logits = x.reshape((batch_size, *self.design_shape))
        log_probs = jax.nn.log_softmax(logits)
        
        if train:
            return distrax.RelaxedOneHotCategorical(temperature=temp, logits=log_probs).sample(seed=jax.random.PRNGKey(0))
        else:
            return jax.nn.one_hot(jnp.argmax(log_probs, axis=-1), self.design_shape[-1])
    
    def sample(self, y, temp=1.0, train=True):
        
        batch_size = y.shape[0]
        z = jax.random.normal(
            jax.random.PRNGKey(0), 
            shape=(batch_size, self.latent_size)
        )
        return self(y, z, temp, train)
    
    
class DiscreteConvGenerator(nn.Module):
    
    design_shape: Sequence[int]  # [seq_length, n_classes]
    latent_size: int
    hidden: int = 50
    
    def setup(self):
        self.embed_0 = nn.Dense(self.hidden)
        
        self.conv_0 = nn.Conv(
            features=self.hidden,
            kernel_size=(3,),
            strides=(1,),
            padding='SAME'
        )
        self.ln_0 = nn.LayerNorm()
        
        self.conv_1 = nn.Conv(
            features=self.hidden,
            kernel_size=(3,),
            strides=(1,),
            padding='SAME'
        )
        self.ln_1 = nn.LayerNorm()
        
        self.conv_2 = nn.Conv(
            features=self.hidden,
            kernel_size=(3,),
            strides=(1,),
            padding='SAME'
        )
        self.ln_2 = nn.LayerNorm()
        
        self.conv_3 = nn.Conv(
            features=self.design_shape[1],  # n_classes
            kernel_size=(3,),
            strides=(1,),
            padding='SAME'
        )
    
    def __call__(self, y, z, temp=1.0, train: bool = True):
        
        y = y.astype(jnp.float32)
        z = z.astype(jnp.float32)
        
        y_embed = self.embed_0(y)  # [batch_size, hidden]
        
        y_embed_expanded = jnp.broadcast_to(
            y_embed[:, None, :],
            (y_embed.shape[0], z.shape[1], y_embed.shape[1])
        )
        
        def apply_conv_block(x, conv, ln, y_embed_expanded):
            x = jnp.concatenate([x, y_embed_expanded], axis=-1)
            x = conv(x)
            x = ln(x)
            return nn.leaky_relu(x, negative_slope=0.2)
        
        x = apply_conv_block(z, self.conv_0, self.ln_0, y_embed_expanded)
        
        x = apply_conv_block(x, self.conv_1, self.ln_1, y_embed_expanded)
        
        x = apply_conv_block(x, self.conv_2, self.ln_2, y_embed_expanded)
        
        x = jnp.concatenate([x, y_embed_expanded], axis=-1)
        logits = self.conv_3(x)
        
        if train:
            return distrax.RelaxedOneHotCategorical(
                temperature=temp,
                logits=jax.nn.log_softmax(logits, axis=-1)
            ).sample(seed=jax.random.PRNGKey(0))
        else:
            return jax.nn.one_hot(
                jnp.argmax(logits, axis=-1),
                self.design_shape[1]
            )
    
    def sample(self, y, temp=1.0, train=True):
        
        batch_size = y.shape[0]
        z = jax.random.normal(
            jax.random.PRNGKey(0),
            shape=(batch_size, self.design_shape[0], self.latent_size)
        )
        return self(y, z, temp, train)
    

class ContinuousGenerator(nn.Module):
    
    design_shape: Sequence[int] 
    latent_size: int           
    hidden: int = 50         
    
    def setup(self):
        self.embed_0 = nn.Dense(self.hidden)
        
        # 主干网络层
        self.dense_0 = nn.Dense(self.hidden)
        self.ln_0 = nn.LayerNorm()
        
        self.dense_1 = nn.Dense(self.hidden)
        self.ln_1 = nn.LayerNorm()
        
        self.dense_2 = nn.Dense(self.hidden)
        self.ln_2 = nn.LayerNorm()
        
        self.dense_3 = nn.Dense(np.prod(self.design_shape))
    
    def __call__(self, y, z=None, train: bool = True):
        
        if z is None:
            key = jax.random.PRNGKey(0)
            z = jax.random.normal(key, shape=(y.shape[0], self.latent_size))
        
        z = z.astype(jnp.float32)
        y = y.astype(jnp.float32)
        
        y_embed = self.embed_0(y)
        
        def apply_dense_block(x, dense, ln, y_embed):
            x = jnp.concatenate([x, y_embed], axis=1)
            x = dense(x)
            x = ln(x)
            return nn.leaky_relu(x, negative_slope=0.2)
        
        x = apply_dense_block(z, self.dense_0, self.ln_0, y_embed)
        x = apply_dense_block(x, self.dense_1, self.ln_1, y_embed)
        x = apply_dense_block(x, self.dense_2, self.ln_2, y_embed)
        
        x = jnp.concatenate([x, y_embed], axis=1)
        x = self.dense_3(x)
        
        return x.reshape((y.shape[0], *self.design_shape))
    
    def sample(self, y, seed=0):
        
        key = jax.random.PRNGKey(seed)
        z = jax.random.normal(
            key,
            shape=(y.shape[0], self.latent_size)
        )
        return self(y, z)    


class ContinuousConvGenerator(nn.Module):
    
    design_shape: Sequence[int]  
    latent_size: int           
    hidden: int = 50          
    
    def setup(self):
        self.embed_0 = nn.Dense(self.hidden)
        
        self.conv_0 = nn.Conv(
            features=self.hidden,
            kernel_size=(3,),
            strides=(1,),
            padding='SAME'
        )
        self.ln_0 = nn.LayerNorm()
        
        self.conv_1 = nn.Conv(
            features=self.hidden,
            kernel_size=(3,),
            strides=(1,),
            padding='SAME'
        )
        self.ln_1 = nn.LayerNorm()
        
        self.conv_2 = nn.Conv(
            features=self.hidden,
            kernel_size=(3,),
            strides=(1,),
            padding='SAME'
        )
        self.ln_2 = nn.LayerNorm()
        
        self.conv_3 = nn.Conv(
            features=self.design_shape[1],
            kernel_size=(3,),
            strides=(1,),
            padding='SAME'
        )
    
    def __call__(self, y, z=None, train: bool = True):
        
        if z is None:
            key = jax.random.PRNGKey(0)
            z = jax.random.normal(
                key,
                shape=(y.shape[0], self.design_shape[0], self.latent_size)
            )
        
        z = z.astype(jnp.float32)
        y = y.astype(jnp.float32)
        
        y_embed = self.embed_0(y)
        
        def broadcast_and_concat(x, y_embed):
            y_broadcasted = jnp.broadcast_to(
                y_embed[:, None, :],
                (y_embed.shape[0], x.shape[1], y_embed.shape[1])
            )
            return jnp.concatenate([x, y_broadcasted], axis=-1)
        
        def apply_conv_block(x, conv, ln, y_embed):
            x = broadcast_and_concat(x, y_embed)
            x = conv(x)
            x = ln(x)
            return nn.leaky_relu(x, negative_slope=0.2)
        
        x = apply_conv_block(z, self.conv_0, self.ln_0, y_embed)
        x = apply_conv_block(x, self.conv_1, self.ln_1, y_embed)
        x = apply_conv_block(x, self.conv_2, self.ln_2, y_embed)
        
        x = broadcast_and_concat(x, y_embed)
        x = self.conv_3(x)
        
        return x
    
    def sample(self, y, seed=0):
        
        key = jax.random.PRNGKey(seed)
        z = jax.random.normal(
            key,
            shape=(y.shape[0], self.design_shape[0], self.latent_size)
        )
        return self(y, z)

