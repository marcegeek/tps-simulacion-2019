import abc

import numpy as np
import matplotlib.pyplot as plt

from common import RandomPopulation


class ProbabilityDistribution(abc.ABC):

    def __init__(self, seed=None):
        self._discrete = False
        self.state = np.random.RandomState(seed)
        self.generator = self._generator()

    def is_continuous(self):
        return not self.is_discrete()

    def is_discrete(self):
        return self._discrete

    def plot_theoretical(self):
        if self.is_continuous():
            x = np.linspace(*self._value_range(), 1000)
            plt.plot(x, self._theoretical_distribution(x))
        else:
            x = np.arange(self._value_range()[0], self._value_range()[1] + 1)
            plt.plot(x, self._theoretical_distribution(x), 'o')

    def random_sample(self, size=None):
        if size is not None:
            return np.array([next(self.generator) for _ in range(size)])
        return next(self.generator)

    @abc.abstractmethod
    def _value_range(self):
        return 0., 1.

    @abc.abstractmethod
    def _theoretical_distribution(self, x):
        return np.array([1.] * len(x))

    def _generator(self):
        while True:
            for r in self._distribution_values():
                yield r

    @abc.abstractmethod
    def _distribution_values(self):
        return [self.state.random_sample()]


class UniformDistribution(ProbabilityDistribution):

    def __init__(self, a, b, seed=None):
        super().__init__(seed=seed)
        self.a = a
        self.b = b

    def _value_range(self):
        return self.a, self.b

    def _theoretical_distribution(self, x):
        return [1. / (self.b - self.a)] * len(x)

    def _distribution_values(self):
        u = self.state.random_sample()
        r = self.a + (self.b - self.a) * u
        return [r]


class ExponentialDistribution(ProbabilityDistribution):

    def __init__(self, alpha, seed=None):
        super().__init__(seed=seed)
        self.alpha = alpha

    def _value_range(self):
        # cubrir el 99.9% de los valores
        return 0, 7. / self.alpha

    def _theoretical_distribution(self, x):
        return self.alpha * np.exp(-self.alpha * x)

    def _distribution_values(self):
        u = self.state.random_sample()
        r = -1. / self.alpha * np.log(u)
        return [r]


class NormalDistribution(ProbabilityDistribution):

    def __init__(self, mu, sigma, seed=None):
        super().__init__(seed=seed)
        self.mu = mu
        self.sigma = sigma

    def _value_range(self):
        # cubrir el 99.99% de los valores
        delta = 4 * self.sigma
        return self.mu - delta, self.mu + delta

    def _theoretical_distribution(self, x):
        return 1 / np.sqrt(2 * np.pi * self.sigma ** 2) * np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2))

    def _distribution_values(self):
        u1 = self.state.random_sample()
        u2 = self.state.random_sample()
        r = np.sqrt(-2 * np.log(u1))
        z0, z1 = r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)
        x0 = z0 * self.sigma + self.mu
        x1 = z1 * self.sigma + self.mu
        return [x0, x1]


class EmpiricalDistribution(ProbabilityDistribution):

    def __init__(self, distribution, seed=None):
        super().__init__(seed=seed)
        self.distribution = distribution
        self._discrete = self.distribution.is_discrete()
        cumulative = self.distribution.cumulative_distribution()
        # agrego frecuencia ficticia (siempre menor que r)
        # para facilitar la generación cuando cae en la
        # primera clase/valor
        self.cumulative = np.insert(cumulative, 0, -1.)

    def _value_range(self):
        first = self.distribution[0]
        last = self.distribution[-1]
        if self.is_continuous():
            return first.val[0], last.val[1]
        return first.val, last.val

    def _theoretical_distribution(self, x):
        arr = np.zeros(len(x))
        if not self.is_continuous():
            freqs = self.distribution.frequencies
        for i in range(len(x)):
            arr[i] = self.distribution.get_freq(x[i])
        return arr

    def _distribution_values(self):
        r = self.state.random_sample()
        for i in range(len(self.cumulative) - 1):
            if self.cumulative[i] < r <= self.cumulative[i + 1]:  # encontrada la clase/valor
                v = self.distribution[i].val
                if self.is_continuous():
                    r = self.state.random_sample()  # a ver donde cae dentro del rango
                    return [v[0] + (v[1] - v[0]) * r]
                else:
                    return [v]


class RandomDistributionPopulation(RandomPopulation):

    def __init__(self, distribution, n=1000):
        self.distribution = distribution
        super().__init__(self.distribution.random_sample(n))

    def plot_theoretical(self):
        self.distribution.plot_theoretical()


# clases envoltorio de conveniencia

class UniformRandomPopulation(RandomDistributionPopulation):

    def __init__(self, a, b, n=1000, seed=None):
        super().__init__(UniformDistribution(a, b, seed=seed),
                         n=n)


class ExponentialRandomPopulation(RandomDistributionPopulation):

    def __init__(self, alpha, n=1000, seed=None):
        super().__init__(ExponentialDistribution(alpha, seed=seed),
                         n=n)


class NormalRandomPopulation(RandomDistributionPopulation):

    def __init__(self, mu, sigma, n=1000, seed=None):
        super().__init__(NormalDistribution(mu, sigma, seed=seed),
                         n=n)


class EmpiricalRandomPopulation(RandomDistributionPopulation):

    def __init__(self, distribution, n=1000, seed=None):
        super().__init__(EmpiricalDistribution(distribution, seed=seed),
                         n=n)
