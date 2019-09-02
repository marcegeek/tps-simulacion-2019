import abc

import numpy as np


class RandomNumberGenerator(abc.ABC):
    """
    Clase base de los generadores de números aleatorios.

    Con interfaces principalmente basadas en las de NumPy.
    """

    def __init__(self, seed=None):
        self._z = 0
        self.seed(seed=seed)

    def random_sample(self, shape=None):
        if shape is None:
            return self._rand()
        arr = np.zeros(shape, dtype='float64')
        for i in range(arr.size):  # recorrer linealmente
            arr.itemset(i, self._rand())
        return arr

    def rand(self, *shape):
        if len(shape) == 0:
            return self.random_sample()
        return self.random_sample(shape)

    def randint(self, low, high=None, shape=None, dtype='l'):
        if high is None:
            low, high = 0, low
        if shape is None:
            return self._randint(low, high)
        arr = np.zeros(shape, dtype=dtype)
        for i in range(arr.size):  # recorrer linealmente
            arr.itemset(i, self._randint(low, high))
        return arr

    @abc.abstractmethod
    def next(self):
        self._z = 0
        return self._z

    @abc.abstractmethod
    def max(self):
        return 0

    def seed(self, seed=None):
        if seed is None:
            self._z = self._gen_seed()
        else:
            if not self._is_seed_valid(seed):
                raise Exception('Invalid seed')
            self._z = seed

    def _is_seed_valid(self, seed):
        return 0 <= seed <= self.max()

    def _gen_seed(self):
        return np.random.randint(0, self.max() + 1)

    def _rand(self):
        return self.next() / (self.max() + 1.)

    def _randint(self, low, high):
        diff = high - low
        range_fraction = (self.max() + 1) // diff
        adjusted_range = range_fraction * diff
        r = self.next()
        while r >= adjusted_range:
            r = self.next()
        return low + r // range_fraction


class MiddleSquareGenerator(RandomNumberGenerator):

    def __init__(self, digits=4, seed=None):
        if digits <= 0 or digits % 2 != 0:
            raise Exception('Invalid digits')
        self._digits = digits
        super().__init__(seed=seed)

    def next(self):
        z_squared = self._z ** 2
        first_digits = z_squared // 10 ** (self._digits // 2)
        self._z = first_digits % 10 ** self._digits
        return self._z

    def max(self):
        return 10 ** self._digits - 1


class LinearCongruentialGenerator(RandomNumberGenerator):

    def __init__(self, modulus, multiplier, increment, seed=None):
        self.modulus = modulus
        self.multiplier = multiplier
        self.increment = increment
        super().__init__(seed=seed)

    def next(self):
        self._z = (self.multiplier * self._z + self.increment) % self.modulus
        return self._z

    def max(self):
        return self.modulus - 1


class RanduGenerator(LinearCongruentialGenerator):

    def __init__(self, seed=None):
        super().__init__(2 ** 31, 65539, 0, seed=seed)

    def _is_seed_valid(self, seed):
        # la semilla tiene que ser impar
        return super()._is_seed_valid(seed) and seed % 2 != 0

    def _gen_seed(self):
        seed = super()._gen_seed()
        while not self._is_seed_valid(seed):
            seed = super()._gen_seed()
        return seed
