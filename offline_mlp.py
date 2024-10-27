import os 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import design_bench as db 
from design_bench.task import Task

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from typing import Sequence
import numpy as np

class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x, training: bool = False):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.leaky_relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x

def create_train_state(rng, learning_rate, input_shape, model):
    params = model.init(rng, jnp.ones(input_shape))
    tx = optax.adamw(learning_rate, weight_decay=1e-5)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        preds = state.apply_fn(params, batch_x)
        return jnp.mean(optax.l2_loss(preds, batch_y))
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_epoch(state, rng, x, y, batch_size):
    train_losses = []
    num_batches = len(x) // batch_size
    
    for i in range(0, len(x), batch_size):
        batch_rng, rng = jax.random.split(rng)
        batch_idx = jax.random.permutation(batch_rng, len(x))[:batch_size]
        batch_x = x[batch_idx]
        batch_y = y[batch_idx]
        
        state, loss = train_step(state, batch_x, batch_y)
        train_losses.append(loss)
    
    return state, jnp.mean(jnp.array(train_losses))

def get_top_solutions(x, y, k=128):
    indices = np.argsort(y.flatten())[-k:]  
    return x[indices], y[indices]



if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    
    task: Task = db.make("Superconductor-RandomForest-v0")
    
    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
    dataset = SuperconductorDataset()
    # from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    # dataset = DKittyMorphologyDataset() 
    y_min = dataset.y.min()
    y_max = dataset.y.max()

    x = task.x.copy()
    y = task.y.copy() 

    x = task.normalize_x(x)
    y = task.normalize_y(y)

    model = MLP(features=[x.shape[1], 2048, 2048, 1])
    learning_rate = 3e-4
    num_epochs = 200
    batch_size = 128 
    
    state = create_train_state(rng, learning_rate, x.shape[1:], model)
    for epoch in range(num_epochs):
        rng, epoch_rng = jax.random.split(rng)
        state, epoch_loss = train_epoch(state, epoch_rng, x, y, batch_size)
        
        print(f'Epoch {epoch}, Loss: {epoch_loss}')
    
    x0, y0 = get_top_solutions(x, y, k=128)
    
    
    @jax.jit
    def predict(params, x):
        return model.apply(params, x)
    
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(x0)

    @jax.jit
    def optimization_step_adam(params, opt_state, x):
        def objective(x):
            return -predict(params, x).sum()  # 负号是因为我们要最大化
        
        grad_fn = jax.grad(objective)
        grads = grad_fn(x)
        updates, opt_state = optimizer.update(grads, opt_state, x)
        x = optax.apply_updates(x, updates)
        return x, opt_state

    # 优化循环
    n_steps = 200
    x_opt = x0

    for step in range(n_steps):
        x_opt, opt_state = optimization_step_adam(state.params, opt_state, x_opt)
        
        # 每100步打印当前预测值
        if step % 10 == 0:
            current_pred = predict(state.params, x_opt)
            test_x = task.denormalize_x(x_opt)
            score_ = task.predict(test_x)
            print(f'Step {step}, Predicted value: {current_pred.mean()}, Score: {(jnp.max(score_) - y_min) / (y_max - y_min)}')
    