import TP2
from common import render_plot


def main():
    rnd = TP2.UniformRandomPopulation(10, 35, n=1000, seed=228547833)
    rnd.histogram()
    rnd.plot_theoric()
    render_plot()
    rnd = TP2.ExponentialRandomPopulation(3, n=1000, seed=1957336057)
    rnd.histogram()
    rnd.plot_theoric()
    render_plot()
    rnd = TP2.NormalRandomPopulation(3, 2, n=1000, seed=1085907535)
    rnd.histogram()
    rnd.plot_theoric()
    render_plot()


if __name__ == '__main__':
    main()
