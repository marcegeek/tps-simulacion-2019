import abc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz


class RandomPopulation(abc.ABC):

    def __init__(self, n=1000, seed=None):
        # fijar semilla aleatoria para asegurar la repetibilidad
        self.generator = np.random.RandomState(seed)
        self._population = self.generator.random_sample(n)
        self._generated = False
        self._classes = None

    @property
    def population(self):
        if not self._generated:
            self._population = self._gen_distribution(self._population)
            self._generated = True
        return self._population

    @abc.abstractmethod
    def _gen_distribution(self, population):
        return population

    @abc.abstractmethod
    def theoric_distribution(self, x):
        return [1.] * len(x)

    @abc.abstractmethod
    def x_range(self):
        # rango empírico
        return min(self.population), max(self.population)

    @property
    def classes(self):
        if self._classes is None:
            n = len(self.population)
            sq = np.sqrt(n)
            self._classes = int(np.ceil(sq))
        return self._classes

    def histogram(self, latexfile=None, standalone_latex=False):
        plt.cla()
        plt.hist(self.population, bins=self.classes, density=True)
        x = np.linspace(*self.x_range(), 1000)
        plt.plot(x, self.theoric_distribution(x))
        plt.grid()
        if latexfile is None:
            plt.show()
        else:
            matplotlib2tikz.save(latexfile,
                                 extra_axis_parameters=[
                                     'scaled ticks=false',
                                     'xticklabel style={/pgf/number format/.cd,fixed,precision=2}',
                                     'yticklabel style={/pgf/number format/.cd,fixed,precision=2}'
                                 ],
                                 standalone=standalone_latex)


class UniformRandomPopulation(RandomPopulation):

    def __init__(self, a, b, n=1000, seed=None):
        super().__init__(n=n, seed=seed)
        self.a = a
        self.b = b

    def _gen_distribution(self, population):
        return self.a + (self.b - self.a) * population

    def theoric_distribution(self, x):
        return [1. / (self.b - self.a)] * len(x)

    def x_range(self):
        return self.a, self.b


class ExponentialRandomPopulation(RandomPopulation):

    def __init__(self, alpha, n=1000, seed=None):
        super().__init__(n=n, seed=seed)
        self.alpha = alpha

    def _gen_distribution(self, population):
        return -1. / self.alpha * np.log(population)

    def theoric_distribution(self, x):
        return self.alpha * np.exp(-self.alpha * x)

    def x_range(self):
        return 0, max(5. / self.alpha, max(self.population))


class NormalRandomPopulation(RandomPopulation):

    def __init__(self, mu, sigma, n=1000, seed=None):
        super().__init__(n=n, seed=seed)
        self.mu = mu
        self.sigma = sigma

    def _gen_distribution(self, population):
        couples = np.split(population, len(population) // 2)
        dist = []
        for u1, u2 in couples:
            assert isinstance(u1, float)
            assert isinstance(u2, float)
            r = np.sqrt(-2 * np.log(u1))
            z0, z1 = r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)
            x0 = z0 * self.sigma + self.mu
            x1 = z1 * self.sigma + self.mu
            dist.append(x0)
            dist.append(x1)
        return dist

    def theoric_distribution(self, x):
        return 1 / np.sqrt(2 * np.pi * self.sigma ** 2, ) * np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2))

    def x_range(self):
        delta = 5 * self.sigma
        return self.mu - delta, self.mu + delta
