from gymnasium.spaces import Space
import numpy as np
from typing import Optional, List
class Permutation(Space):
    def __init__(self, n: int, start: int = 0):
        self.n = n
        self._shape = (n,)
        self.dtype = np.int32
        self.start = start
        super().__init__(shape=self._shape, dtype=self.dtype)

    def sample(self):
        return np.random.permutation(self.n).astype(self.dtype) + self.start

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if x.shape != self._shape:
            return False
        if not np.issubdtype(x.dtype, np.integer):
            return False
        # need to account for start 
        return np.array_equal(np.sort(x), np.arange(self.start, self.start + self.n))

    def __repr__(self):
        return f"Permutation({self.n})"

    def __eq__(self, other):
        return isinstance(other, Permutation) and self.n == other.n
