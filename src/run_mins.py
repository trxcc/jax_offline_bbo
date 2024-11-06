from typing import Optional, Tuple, Any, Dict, List, Callable, Union

import hydra
import rootutils
import wandb 
import jax 
import jax.numpy as jnp
import flax.linen as nn 
from omegaconf import OmegaConf, DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.logger import RankedLogger
from src.utils.utils import task_wrapper, seed_everything, get_metric_value, obtain_percentile_score
from src.data.datamodule import JAXDataModule
from src.logger.base_logger import BaseLogger
from src.trainer.mlp_trainer import Trainer
from src.search.base_searcher import Searcher
from src.task.base_task import OfflineBBOExperimenter
from src.model.gan_component import (
    Discriminator,
    ConvDiscriminator,
    DiscreteGenerator,
    DiscreteConvGenerator,
    ContinuousGenerator,
    ContinuousConvGenerator
)
from src.trainer.mins.weight_gan_trainer import WeightGANTrainer
from src.trainer.mins.utils import (
    get_weights
)
from src._typing import PRNGKeyArray as KeyArray

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    
    if cfg.get("seed"):
        key: KeyArray = seed_everything(cfg.seed)
        seed = cfg.seed
    else:
        import random
        seed = random.randint(0, 1e-6)
        key: KeyArray = seed_everything(seed)
    
    log.info(f"Instantiating task <{cfg.task._target_}>")
    task: OfflineBBOExperimenter = hydra.utils.instantiate(cfg.task)
    
    # problem_statement = task.problem_statement()
    # log.info(f"Problem: {problem_statement}")
    
    x_transforms, y_transforms = [], []
    x_restores, y_restores = [], []
    
    if task.require_normalize_ys:
        y_transforms.append(task.normalize_y)
        y_restores.insert(0, task.denormalize_y)
        
    if task.require_to_logits and task.is_discrete:
        x_transforms.append(task.to_logits)
        x_restores.insert(0, task.to_integers)
        
    if task.require_normalize_xs:
        x_transforms.append(task.normalize_x)
        x_restores.insert(0, task.denormalize_x)

    key, data_key, trainer_key, gan_data_key = jax.random.split(key, num=4)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: JAXDataModule = hydra.utils.instantiate(
        config=cfg.data,
        x_transforms=x_transforms,
        y_transforms=y_transforms,
        x_restores=x_restores,
        y_restores=y_restores
    )
    
    datamodule.setup(
        x=task.x,
        y=task.y,
        x_test=task.x_ood,
        y_test=task.y_ood,
        random_key=data_key,
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: nn.Module = hydra.utils.instantiate(cfg.model, task=task)

    log.info(f"Instantiating logger <{cfg.logger._target_}>")
    logger: BaseLogger = hydra.utils.instantiate(cfg.logger)
    
    log.info(f"Instantiating loss function <{cfg.loss._target_}>")
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = hydra.utils.instantiate(cfg.loss)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config=cfg.trainer, 
        loss_fn=loss_fn,
        data_module=datamodule,
        seed=cfg.seed,
        rng=trainer_key,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "task": task,
        "datamodule": datamodule,
        "model": model,
        "loss_fn": loss_fn,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, input_shape=(datamodule.batch_size, *datamodule.input_shape))
    
    log.info(f"Instantiating GAN-datamodule <{cfg.data._target_}>")
    gan_datamodule: JAXDataModule = hydra.utils.instantiate(
        config=cfg.gan_data,
        x_transforms=x_transforms,
        y_transforms=y_transforms,
        x_restores=x_restores,
        y_restores=y_restores,
    )
    gan_datamodule.setup(
        x=task.x,
        y=task.y,
        w=get_weights(datamodule.y, base_temp=cfg.get("base_temp", None)),
        random_key=gan_data_key,
    )
    
    explore_discriminator: Union[Discriminator, ConvDiscriminator] = hydra.utils.instantiate(
        cfg.discriminator, design_shape=datamodule.input_shape
    )
    
    if task.is_discrete:
        explore_generator: Union[DiscreteGenerator, DiscreteConvGenerator] = hydra.utils.instantiate(
            cfg.discrete_generator, design_shape=datamodule.input_shape
        )
    else:
        explore_generator: Union[ContinuousGenerator, ContinuousConvGenerator] = hydra.utils.instantiate(
            cfg.continuous_generator, design_shape=datamodule.input_shape
        )
    
    explore_gan: WeightGANTrainer = hydra.utils.instantiate(
        cfg.gan_trainer,
        data_module=gan_datamodule,
        is_discrete=task.is_discrete,
        seed=seed,
        rng=key,
        save_prefix="explore_gan",
        logger=logger
    )
    
    exploit_discriminator: Union[Discriminator, ConvDiscriminator] = hydra.utils.instantiate(
        cfg.discriminator, design_shape=datamodule.input_shape
    )
    
    if task.is_discrete:
        exploit_generator: Union[DiscreteGenerator, DiscreteConvGenerator]= hydra.utils.instantiate(
            cfg.discrete_generator, design_shape=datamodule.input_shape
        )
    else:
        exploit_generator: Union[ContinuousGenerator, ContinuousConvGenerator] = hydra.utils.instantiate(
            cfg.continuous_generator, design_shape=datamodule.input_shape
        )
        
    exploit_gan: WeightGANTrainer = hydra.utils.instantiate(
        cfg.gan_trainer,
        data_module=gan_datamodule,
        is_discrete=task.is_discrete,
        seed=seed,
        rng=key,
        save_prefix="exploit_gan",
        logger=logger
    )
    
    exploit_gan.fit(
        exploit_generator, 
        exploit_discriminator,
        input_shape=(gan_datamodule.batch_size, *gan_datamodule.input_shape),
    )
    
    condition_ys = jnp.tile(jnp.max(datamodule.y, keepdims=True), (cfg.get("num_solutions", 128), 1))
    
    if task.is_discrete:
        explore_gan.start_temp = explore_gan.final_temp
        exploit_gan.start_temp = exploit_gan.final_temp
    
    best_model = trainer.load_model()
    metric_dict = trainer.get_history()
    
    
    x_res, _ = datamodule.restore_data(x=x_res)
    
    score_dict = task.score(x_res)
    score = score_dict["Score"]
    normalize_score = score_dict["Normalized_Score"]
    
    score_dict = {**obtain_percentile_score(score, "Score"),
                  **obtain_percentile_score(normalize_score, "Normalized_Score")}
    
    logger.log_metrics(score_dict)
    log.info(score_dict)

    logger.finish()
    return metric_dict, object_dict, score_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]: 
    
    if cfg.get("wandb_api"):
        wandb.login(key=cfg.wandb_api)
        
    metric_dict, _, score_dict = train(cfg)
    
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("test_auprc")
    )
    
    log.info(score_dict)
    
    return metric_value

if __name__ == "__main__":
    main()