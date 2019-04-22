import TP2


def main():
    rnd = TP2.RandomPopulation(1000)
    rnd.plot(TP2.UNIFORM, 10, 35)
    rnd.plot(TP2.EXPONENTIAL, 2)


if __name__ == '__main__':
    main()
