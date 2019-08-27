import numpy as np


class RandomPopulation:

    def __init__(self, population):
        self._population = np.array(population)

    @property
    def values(self):
        return self._population.copy()

    @property
    def classes(self):
        n = len(self._population)
        sq = np.sqrt(n)
        return int(np.ceil(sq))

    def is_continuous(self):
        return not self.is_discrete()

    def is_discrete(self):
        return self._population.dtype.kind == 'i'

    def frequency_distribution(self):
        return FrequencyDistribution(population=self)

    def mean(self):
        return self._population.mean()

    def variance(self):
        return self._population.var()

    def standard_deviation(self):
        return self._population.std()

    def plot_histogram(self, axes, label=None):
        if self.is_continuous():
            axes.hist(self._population, bins=self.classes, density=True, label=label)
        else:
            dist = self.frequency_distribution()
            # tomar y avanzar el color
            points, = axes.plot(dist.values, dist.frequencies, ',')
            axes.vlines(dist.values, 0, dist.frequencies, lw=10, colors=points.get_color(), label=label)
