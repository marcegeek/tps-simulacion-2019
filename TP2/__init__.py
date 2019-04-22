import numpy as np

import matplotlib.pyplot as plt

UNIFORM, EXPONENTIAL, NORMAL = range(1, 3 + 1)


class RandomPopulation:
    SEED = 3619001274

    def __init__(self, n):
        # fijar semilla aleatoria para asegurar la repetibilidad
        np.random.seed(self.SEED)
        self.uniform_base = np.random.sample(n)
        self._bins = None

    def map_methods(self, method):
        if method == UNIFORM:
            return self.uniform_random, self.uniform_pdf, lambda pop, a, b: (a, b)
        elif method == EXPONENTIAL:
            return self.exponential_random, self.exponential_pdf, self.exponential_xrange

    @property
    def bins(self):
        if self._bins is None:
            n = len(self.uniform_base)
            sq = np.sqrt(n)
            self._bins = int(np.ceil(sq))
        return self._bins

    @staticmethod
    def uniform_pdf(x, a, b):
        return [1. / (b - a)] * len(x)

    def uniform_random(self, a, b):
        return a + (b - a) * self.uniform_base

    @staticmethod
    def exponential_pdf(x, alpha):
        return alpha * np.exp(-alpha * x)

    @staticmethod
    def exponential_xrange(pop, alpha):
        return 0, max(5. / alpha, max(pop))

    def exponential_random(self, alpha):
        return -1. / alpha * np.log(self.uniform_base)

    def normal_random(self):
        pass

    def plot(self, method, *args):
        rnd, pdf, rng = self.map_methods(method)
        res = rnd(*args)
        plt.hist(res, self.bins, density=True)
        min_x, max_x = rng(res, *args)
        x = np.linspace(min_x, max_x, np.round((max_x - min_x) * 10))
        theorical = pdf(x, *args)
        plt.plot(x, theorical)
        plt.grid()
        plt.show()
