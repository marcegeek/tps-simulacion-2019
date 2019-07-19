import numpy as np
import matplotlib.pyplot as plt


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

    def plot_histogram(self):
        if self.is_continuous():
            plt.hist(self._population, bins=self.classes, density=True)
        else:
            dist = self.frequency_distribution()
            # tomar y avanzar el color
            ax, = plt.plot(dist.values, dist.frequencies, ',')
            plt.vlines(dist.values, 0, dist.frequencies, lw=10, colors=ax.get_color())
        plt.grid()
