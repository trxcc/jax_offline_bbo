from typing import Optional, Tuple, Any, Dict, List, Callable

import hydra
import rootutils
import wandb 
import jax.numpy as jnp
import flax.linen as nn 
from omegaconf import OmegaConf, DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.logger import RankedLogger
from src.utils.utils import task_wrapper, seed_everything, get_metric_value, obtain_percentile_score
from src.data.datamodule import JAXDataModule
from src.logger.base_logger import BaseLogger
from src.trainer.base_trainer import Trainer
from src.search.base_searcher import Searcher
from src.task.base_task import OfflineBBOExperimenter
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
        random_key=key
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: nn.Module = hydra.utils.instantiate(cfg.model, input_size=datamodule.input_size)

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
        rng=key,
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

    
    best_model = trainer.load_model()
    metric_dict= trainer.get_history()
    
    log.info(f"Instantiating searcher <{cfg.search._target_}>")
    searcher: Searcher = hydra.utils.instantiate(
        cfg.search, 
        score_fn=lambda x: trainer.predict(x, params=best_model.params),
        datamodule=datamodule,
        task=task
    )
    object_dict["searcher"] = searcher
    x_res = searcher.run()
    
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