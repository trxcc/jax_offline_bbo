import numpy as np
import jax 
import jax.numpy as jnp
from typing import Callable, Optional, Sequence, Tuple, Union, Type

from src.utils.utils import train_val_split
from src._typing import PRNGKeyArray as KeyArray

class JAXDataModule:
    def __init__(
        self,
        batch_size: int,
        val_split: float = 0.2,
        x_transforms: Optional[Sequence[
            Callable[[jnp.ndarray], jnp.ndarray]]] = None,
        y_transforms: Optional[Sequence[
            Callable[[jnp.ndarray], jnp.ndarray]]] = None,
        x_restores: Optional[Sequence[
            Callable[[jnp.ndarray], jnp.ndarray]]] = None,
        y_restores: Optional[Sequence[
            Callable[[jnp.ndarray], jnp.ndarray]]] = None,
    ) -> None:
        self.batch_size = batch_size
        self.val_split = val_split
        
        self.x_transforms = x_transforms or []
        self.y_transforms = y_transforms or []
        
        self.x_restores = x_restores or []
        self.y_restores = y_restores or []
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def transform_data(
        self, 
        x: Optional[Union[np.ndarray, jnp.ndarray]] = None, 
        y: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray]:
        if isinstance(x, np.ndarray):
            x = jnp.array(x)
        if isinstance(y, np.ndarray):
            y = jnp.array(y)
        
        if x is not None:
            for transform in self.x_transforms:
                x = transform(x)
        if y is not None:
            for transform in self.y_transforms:
                y = transform(y)
        return x, y
    
    def restore_data(
        self, 
        x: Optional[Union[np.ndarray, jnp.ndarray]] = None, 
        y: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray]:
        if isinstance(x, np.ndarray):
            x = jnp.array(x)
        if isinstance(y, np.ndarray):
            y = jnp.array(y)
        
        if x is not None:
            for restore in self.x_restores:
                x = restore(x)
        if y is not None:
            for restore in self.y_restores:
                y = restore(y)
        return x, y
    
    def setup(
        self,
        x: Union[np.ndarray, jnp.ndarray],
        y: Union[np.ndarray, jnp.ndarray],
        x_test: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        y_test: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        w: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        random_key: Optional[KeyArray] = None,
    ):
        self.eval_test = x_test is not None and y_test is not None 
        
        x, y = self.transform_data(x, y)
        if self.eval_test:
            x_test, y_test = self.transform_data(x_test, y_test)
        
        self._x, self._y = x, y 
        self._x_test, self._y_test = x_test, y_test
        self.has_w = w is not None 
        
        if w is not None:
            x_train, y_train, x_val, y_val, w_train = train_val_split(
                x, y, w=w,
                val_size=self.val_split,
                key=random_key
            )
        else:
            x_train, y_train, x_val, y_val = train_val_split(
                x, y,
                val_size=self.val_split,
                key=random_key
            )
        
        self.train_data = (jnp.array(x_train), jnp.array(y_train), jnp.array(w_train)) \
            if w is not None else (jnp.array(x_train), jnp.array(y_train))
        self.val_data = (jnp.array(x_val), jnp.array(y_val))
        
        if self.eval_test:
            self.test_data = (jnp.array(x_test), jnp.array(y_test))
            
    @property
    def input_size(self) -> int:
        assert self.train_data is not None, "please set up date module first"
        return np.prod(self.train_data[0].shape[1:]).item()
    
    @property
    def input_shape(self) -> Tuple[int]:
        assert self.train_data is not None, "please set up date module first"
        return tuple(self.train_data[0].shape[1:])
    
    @property
    def input_dtype(self) -> Type:
        assert self.train_data is not None, "please set up date module first"
        return self.train_data[0].dtype
    
    @property
    def x(self) -> jnp.ndarray:
        assert self.train_data is not None, "please set up date module first"
        return self._x
    
    @property
    def y(self) -> jnp.ndarray:
        assert self.train_data is not None, "please set up date module first"
        return self._y
    
    @property
    def x_test(self) -> jnp.ndarray:
        assert self.train_data is not None, "please set up date module first"
        assert self.eval_test, "test data is not prepared"
        return self._x
    
    @property
    def y_test(self) -> jnp.ndarray:
        assert self.train_data is not None, "please set up date module first"
        assert self.eval_test, "test data is not prepared"
        return self._y
        
    def get_batch(self, key: KeyArray, data: Tuple[jnp.ndarray], is_val: bool = False):
        if self.has_w and not is_val:
            x, y, w = data
        else:
            x, y = data 
        num_samples = len(x)
        
        shuffled_idx = jax.random.permutation(key, num_samples)
        num_batches = num_samples // self.batch_size
        
        shuffled_idx = shuffled_idx[:num_batches * self.batch_size]
        batch_idx = shuffled_idx.reshape((num_batches, self.batch_size))
        
        x_batches = x[batch_idx]
        y_batches = y[batch_idx]
        
        if self.has_w and not is_val:
            w_batches = w[batch_idx]
            return x_batches, y_batches, w_batches 
    
        return x_batches, y_batches
    
    def train_dataloader(self, key: KeyArray):
        return self.get_batch(key, self.train_data)
    
    def val_dataloader(self, key: KeyArray):
        return self.get_batch(key, self.val_data, is_val=True)
    
    def test_dataloader(self, key: KeyArray):
        return self.get_batch(key, self.test_data, is_val=True) if self.eval_test else None 