import numpy as np
import matplotlib.pyplot as plt


class RandomPopulation:

    def __init__(self, population):
        self._population = np.array(population)

    @property
    def population(self):
        return self._population.copy()

    @property
    def classes(self):
        n = len(self._population)
        sq = np.sqrt(n)
        return int(np.ceil(sq))

    def plot_histogram(self):
        plt.hist(self._population, bins=self.classes, density=True)
        plt.grid()
