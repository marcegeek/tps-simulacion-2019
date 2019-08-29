from code import stats, plotter
import code.stats.distributions as dist


def make_plots(rnd_pop, **render_args):
    fig = plotter.SimpleFigure(xlabel='Valores', ylabel='Probabilidades')
    rnd_pop.plot_histogram(axes=fig.ax, label='Distribución generada')
    rnd_pop.plot_theoretical(axes=fig.ax, label='Distribución teórica')
    fig.render(**render_args)


def main():
    pop = dist.UniformRandomPopulation(10, 35, size=1000, seed=228547833)
    make_plots(pop, latexfile='latex/uniform.tex')

    pop = dist.ExponentialRandomPopulation(3, size=1000, seed=1957336057)
    make_plots(pop, latexfile='latex/exponential.tex')

    pop = dist.GammaRandomPopulation(4, 3, size=1000, seed=1047849975)
    make_plots(pop, latexfile='latex/gamma.tex')

    pop = dist.NormalRandomPopulation(3, 2, size=1000, seed=1085907535)
    make_plots(pop, latexfile='latex/normal.tex')

    pop = dist.BinomialRandomPopulation(10, 0.7, size=1000, seed=2961379677)
    make_plots(pop, latexfile='latex/binomial.tex')

    pop = dist.PoissonRandomPopulation(4, size=1000, seed=2061262339)
    make_plots(pop, latexfile='latex/poisson.tex')

    pop = dist.EmpiricalRandomPopulation(stats.FrequencyDistribution(
        distr=[
            stats.FrequencyDistributionEntry((1, 2), 1),
            stats.FrequencyDistributionEntry((2, 3), 2),
            stats.FrequencyDistributionEntry((3, 4), 5),
            stats.FrequencyDistributionEntry((4, 5), 4),
            stats.FrequencyDistributionEntry((5, 6), 2),
            stats.FrequencyDistributionEntry((6, 7), 1)
        ]),
        size=1000, seed=3131638626)
    make_plots(pop, latexfile='latex/empirical-cont.tex')

    pop = dist.EmpiricalRandomPopulation(stats.FrequencyDistribution(
        distr=[
            stats.FrequencyDistributionEntry(2, 3),
            stats.FrequencyDistributionEntry(3, 2),
            stats.FrequencyDistributionEntry(4, 5),
            stats.FrequencyDistributionEntry(5, 8),
            stats.FrequencyDistributionEntry(6, 4),
            stats.FrequencyDistributionEntry(7, 1)
        ]),
        size=1000, seed=2662949332)
    make_plots(pop, latexfile='latex/empirical-disc.tex')


if __name__ == '__main__':
    main()
