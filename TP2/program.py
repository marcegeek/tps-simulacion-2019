import TP2


def main():
    rnd = TP2.UniformRandomPopulation(10, 35, n=1000, seed=228547833)
    rnd.histogram(latexfile='tex/uniform.tex')
    rnd = TP2.ExponentialRandomPopulation(3, n=1000, seed=1957336057)
    rnd.histogram(latexfile='tex/exponential.tex')
    rnd = TP2.NormalRandomPopulation(3, 2, n=1000, seed=1085907535)
    rnd.histogram(latexfile='tex/normal.tex')


if __name__ == '__main__':
    main()
