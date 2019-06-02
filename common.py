import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz


class RandomPopulation:

    def __init__(self, population):
        self._population = population

    @property
    def population(self):
        return self._population

    @property
    def classes(self):
        n = len(self.population)
        sq = np.sqrt(n)
        return int(np.ceil(sq))

    def histogram(self):
        plt.hist(self.population, bins=self.classes, density=True)
        plt.grid()


def render_plot(latexfile=None, standalone_latex=False):
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
