import os 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from typing import Sequence, Any
from flax import serialization  # 用于保存/加载模型
import numpy as np

class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x, training: bool = False):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x

def create_train_state(rng, learning_rate, input_shape, model):
    params = model.init(rng, jnp.ones(input_shape))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        preds = state.apply_fn(params, batch_x)
        return jnp.mean((preds - batch_y) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

if __name__ == "__main__":
    # 设置随机种子
    rng = jax.random.PRNGKey(0)

    # 创建一些示例数据
    x = np.random.normal(size=(100, 10))  # 100个样本，每个10维
    y = np.random.normal(size=(100, 1))   # 100个目标值，每个1维

    # 模型配置
    model = MLP(features=[64, 32, 1])  # 两个隐层(64和32个节点)，输出1维
    learning_rate = 0.001
    num_epochs = 100

    # 初始化训练状态
    state = create_train_state(rng, learning_rate, x.shape[1:], model)

    # 训练循环
    for epoch in range(num_epochs):
        state, loss = train_step(state, x, y)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # 预测
    @jax.jit
    def predict(state, x):
        return state.apply_fn(state.params, x)

    test_x = np.random.normal(size=(5, 10))
    predictions = predict(state, test_x)
    print("Test predictions:", predictions)