import TP2
from common import plotter


def main():
    rnd = TP2.UniformRandomPopulation(10, 35, n=1000, seed=228547833)
    rnd.plot_histogram()
    rnd.plot_theoretical()
    plotter.render()
    rnd = TP2.ExponentialRandomPopulation(3, n=1000, seed=1957336057)
    rnd.plot_histogram()
    rnd.plot_theoretical()
    plotter.render()
    rnd = TP2.NormalRandomPopulation(3, 2, n=1000, seed=1085907535)
    rnd.plot_histogram()
    rnd.plot_theoretical()
    plotter.render()


if __name__ == '__main__':
    main()
