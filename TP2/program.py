import TP2
import common
from common import plotter


def makeplots(rnd_pop, **render_args):
    fig = plotter.SimpleFigure(xlabel='Valores', ylabel='Probabilidades')
    rnd_pop.plot_histogram(axes=fig.ax, label='Distribución generada')
    rnd_pop.plot_theoretical(axes=fig.ax, label='Distribución teórica')
    fig.render(**render_args)


def main():
    pop = TP2.UniformRandomPopulation(10, 35, size=1000, seed=228547833)
    makeplots(pop)

    pop = TP2.ExponentialRandomPopulation(3, size=1000, seed=1957336057)
    makeplots(pop)

    pop = TP2.GammaRandomPopulation(4, 3, size=1000, seed=1047849975)
    makeplots(pop)

    pop = TP2.NormalRandomPopulation(3, 2, size=1000, seed=1085907535)
    makeplots(pop)

    pop = TP2.BinomialRandomPopulation(10, 0.7, size=1000, seed=2961379677)
    makeplots(pop)

    pop = TP2.EmpiricalRandomPopulation(common.FrequencyDistribution(
        distr=[
            common.FrequencyDistributionEntry((1, 2), 1),
            common.FrequencyDistributionEntry((2, 3), 2),
            common.FrequencyDistributionEntry((3, 4), 5),
            common.FrequencyDistributionEntry((4, 5), 4),
            common.FrequencyDistributionEntry((5, 6), 2),
            common.FrequencyDistributionEntry((6, 7), 1)
        ]),
        size=1000, seed=3131638626)
    makeplots(pop)

    pop = TP2.EmpiricalRandomPopulation(common.FrequencyDistribution(
        distr=[
            common.FrequencyDistributionEntry(2, 3),
            common.FrequencyDistributionEntry(3, 2),
            common.FrequencyDistributionEntry(4, 5),
            common.FrequencyDistributionEntry(5, 8),
            common.FrequencyDistributionEntry(6, 4),
            common.FrequencyDistributionEntry(7, 1)
        ]),
        size=1000, seed=2662949332)
    makeplots(pop)


if __name__ == '__main__':
    main()
