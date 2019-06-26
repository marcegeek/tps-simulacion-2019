import TP2
import common
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
    rnd = TP2.EmpiricalRandomPopulation(common.FrequencyDistribution(
        distr=[
            common.FrequencyDistributionEntry((1, 2), 1),
            common.FrequencyDistributionEntry((2, 3), 2),
            common.FrequencyDistributionEntry((3, 4), 5),
            common.FrequencyDistributionEntry((4, 5), 4),
            common.FrequencyDistributionEntry((5, 6), 2),
            common.FrequencyDistributionEntry((6, 7), 1)
        ]),
        n=1000, seed=3131638626)
    rnd.plot_histogram()
    rnd.plot_theoretical()
    plotter.render()
    rnd = TP2.EmpiricalRandomPopulation(common.FrequencyDistribution(
        distr=[
            common.FrequencyDistributionEntry(2, 3),
            common.FrequencyDistributionEntry(3, 2),
            common.FrequencyDistributionEntry(4, 5),
            common.FrequencyDistributionEntry(5, 8),
            common.FrequencyDistributionEntry(6, 4),
            common.FrequencyDistributionEntry(7, 1)
        ]),
        n=1000, seed=2662949332)
    rnd.plot_histogram()
    rnd.plot_theoretical()
    plotter.render()


if __name__ == '__main__':
    main()
