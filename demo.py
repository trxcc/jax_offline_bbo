from typing import Optional, Union

import random
import numpy as np 
import jax 
import jax.numpy as jnp
import optax 
from vizier._src.pyvizier.shared.base_study_config import ProblemStatement 

from src.utils.logger import RankedLogger
from src.logger.wandb_logger import WandBLogger
from src.utils.utils import seed_everything
from src.task.base_task import OfflineBBOExperimenter
from src.data.datamodule import JAXDataModule
from src.model.mlp import MLP
from src.trainer.mlp_trainer import MLPTrainer
from src.search.grad import AdamSearcher
from src._typing import PRNGKeyArray as KeyArray

log = RankedLogger(__name__, rank_zero_only=True)


class TestOfflineExperimenter(OfflineBBOExperimenter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_mean = np.mean(self.x_np, axis=0)
        self.x_std = np.std(self.x_np, axis=0)
        self.y_mean = np.mean(self.y_np)
        self.y_std = np.std(self.y_np)
        
    def normalize_x(self, x):
        return (x - self.x_mean) / self.x_std 
    
    def normalize_y(self, y):
        return (y - self.y_mean) / self.y_std 
    
    def denormalize_x(self, x):
        return x * self.x_std + self.x_mean 
    
    def denormalize_y(self, y):
        return y * self.y_std + self.y_mean
    
    def to_integers(self, x):
        pass 
    
    def to_logits(self, x):
        pass 
    
    def problem_statement(self) -> ProblemStatement:
        pass 


def evaluate(x: np.ndarray) -> np.ndarray:
    return np.mean(
        -x**2 + 2*x + 1,
        axis=1
    ).reshape(-1, 1)

seed = random.randint(0, 1e6)
key: KeyArray = seed_everything(seed)
  
lb = -100
ub = 100
n_dim = 2
n_data = 1000
x = np.random.rand(n_data, n_dim) * (ub - lb) + lb 
y = evaluate(x)

log.info("Initializing experimenter...")
task = TestOfflineExperimenter(
    task_name="test_function",
    eval_fn=evaluate,
    x_np=x,
    y_np=y,
    is_discrete=False,
    require_normalize_xs=True,
    require_normalize_ys=True
)

x_transforms = [task.normalize_x]
x_restores = [task.denormalize_x]
y_transforms = [task.normalize_y]
y_restores = [task.denormalize_y]

key, data_key, trainer_key, searcher_key = jax.random.split(key, num=4)

log.info("Initializing datamodule...")
datamodule = JAXDataModule(
    batch_size=128,
    val_split=0.2,
    x_transforms=x_transforms,
    y_transforms=y_transforms,
    x_restores=x_restores,
    y_restores=y_restores
)

datamodule.setup(
    x=task.x,
    y=task.y,
    random_key=data_key
)

log.info("Initializing model...")
model = MLP(
    task=task,
    hidden_sizes=[128, 128],
    output_size=1,
)

log.info("Initializing logger...")
logger = WandBLogger(
    project="test-offline",
)

log.info("Initializing loss function...")
loss_fn = optax.l2_loss

log.info("Initializing trainer...")
trainer = MLPTrainer(
    loss_fn=loss_fn,
    optimizer=optax.adam(learning_rate=3e-4),
    data_module=datamodule,
    max_epochs=5,
    seed=seed,
    rng=trainer_key,
    eval_test=False,
    checkpoint_dir="/data/trx/jax_offline_bbo/checkpoints", # FIXME: Here it should a absolute path, otherwise orbax with raise an error
    logger=logger
)

log.info("Starting training!")
trainer.fit(model=model, input_shape=(datamodule.batch_size, *datamodule.input_shape))

best_model = trainer.load_model()
log.info("Initializing searcher...")
searcher = AdamSearcher(
    key=searcher_key,
    score_fn=lambda x: trainer.predict(x, params=best_model.params),
    datamodule=datamodule,
    task=task,
    num_solutions=128,
    learning_rate=1e-3,
    search_steps=200,
    scale_lr=False
)

x_res = searcher.run()
x_res, _ = datamodule.restore_data(x=x_res)

score_dict = task.score(x_res)
print(score_dict)
