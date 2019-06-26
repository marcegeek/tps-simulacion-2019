import abc

import numpy as np
import matplotlib.pyplot as plt

from common import RandomPopulation


class RandomDistributionGenerator:

    def __init__(self, seed=None):
        self.state = np.random.RandomState(seed)
        self.generator = self._generator()

    def random_sample(self, size=None):
        if size is not None:
            return np.array([next(self.generator) for _ in range(size)])
        return next(self.generator)

    def _generator(self):
        while True:
            for r in self._distribution_values():
                yield r

    def plot_theoretical_distribution(self):
        x = np.linspace(*self._value_range(), 1000)
        plt.plot(x, self._theoretical_distribution(x))

    @abc.abstractmethod
    def _distribution_values(self):
        return [self.state.random_sample()]

    @abc.abstractmethod
    def _value_range(self):
        return 0., 1.

    @abc.abstractmethod
    def _theoretical_distribution(self, x):
        return 1.


class UniformDistributionGenerator(RandomDistributionGenerator):

    def __init__(self, a, b, seed=None):
        super().__init__(seed=seed)
        self.a = a
        self.b = b

    def _distribution_values(self):
        u = self.state.random_sample()
        r = self.a + (self.b - self.a) * u
        return [r]

    def _value_range(self):
        return self.a, self.b

    def _theoretical_distribution(self, x):
        return [1. / (self.b - self.a)] * len(x)


class ExponentialDistributionGenerator(RandomDistributionGenerator):

    def __init__(self, alpha, seed=None):
        super().__init__(seed=seed)
        self.alpha = alpha

    def _distribution_values(self):
        u = self.state.random_sample()
        r = -1. / self.alpha * np.log(u)
        return [r]

    def _value_range(self):
        # cubrir el 99.9% de los valores
        return 0, 7. / self.alpha

    def _theoretical_distribution(self, x):
        return self.alpha * np.exp(-self.alpha * x)


class NormalDistributionGenerator(RandomDistributionGenerator):

    def __init__(self, mu, sigma, seed=None):
        super().__init__(seed=seed)
        self.mu = mu
        self.sigma = sigma

    def _distribution_values(self):
        u1 = self.state.random_sample()
        u2 = self.state.random_sample()
        r = np.sqrt(-2 * np.log(u1))
        z0, z1 = r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)
        x0 = z0 * self.sigma + self.mu
        x1 = z1 * self.sigma + self.mu
        return [x0, x1]

    def _value_range(self):
        # cubrir el 99.99% de los valores
        delta = 4 * self.sigma
        return self.mu - delta, self.mu + delta

    def _theoretical_distribution(self, x):
        return 1 / np.sqrt(2 * np.pi * self.sigma ** 2) * np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2))


class EmpiricalDistributionGenerator(RandomDistributionGenerator):

    def __init__(self, distribution, seed=None):
        super().__init__(seed=seed)
        self.distribution = distribution
        cumulative = self.distribution.cumulative_distribution()
        # agrego frecuencia ficticia (siempre menor que r)
        # para facilitar la generación cuando cae en la
        # primera clase/valor
        self.cumulative = np.insert(cumulative, 0, -1.)

    def _distribution_values(self):
        r = self.state.random_sample()
        for i in range(len(self.cumulative) - 1):
            if self.cumulative[i] < r <= self.cumulative[i + 1]:  # encontrada la clase/valor
                v = self.distribution[i].val
                if self.distribution.is_continuous():
                    r = self.state.random_sample()  # a ver donde cae dentro del rango
                    return [v[0] + (v[1] - v[0]) * r]
                else:
                    return [v]

    def _value_range(self):
        first = self.distribution[0]
        last = self.distribution[-1]
        if self.distribution.is_continuous():
            return first.val[0], last.val[1]
        return first.val, last.val

    def _theoretical_distribution(self, x):
        arr = np.zeros(len(x))
        if not self.distribution.is_continuous():
            return arr  # TODO implementar caso discreto
        for i in range(len(x)):
            arr[i] = self.distribution.get_freq(x[i])
        return arr


class RandomDistributionPopulation(RandomPopulation):

    def __init__(self, generator=None, n=1000, seed=None):
        if generator is None:
            # fijar semilla aleatoria para asegurar la repetibilidad
            self.generator = RandomDistributionGenerator(seed=seed)
        else:
            self.generator = generator
        super().__init__(self.generator.random_sample(n))

    def plot_theoretical(self):
        self.generator.plot_theoretical_distribution()


# clases envoltorio de conveniencia

class UniformRandomPopulation(RandomDistributionPopulation):

    def __init__(self, a, b, n=1000, seed=None):
        super().__init__(generator=UniformDistributionGenerator(a, b, seed=seed),
                         n=n)


class ExponentialRandomPopulation(RandomDistributionPopulation):

    def __init__(self, alpha, n=1000, seed=None):
        super().__init__(ExponentialDistributionGenerator(alpha, seed=seed),
                         n=n)


class NormalRandomPopulation(RandomDistributionPopulation):

    def __init__(self, mu, sigma, n=1000, seed=None):
        super().__init__(NormalDistributionGenerator(mu, sigma, seed=seed),
                         n=n)


class EmpiricalRandomPopulation(RandomDistributionPopulation):

    def __init__(self, distribution, n=1000, seed=None):
        super().__init__(EmpiricalDistributionGenerator(distribution, seed=seed),
                         n=n)
