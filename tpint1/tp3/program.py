import numpy as np

from code import stats
from code.plotter import SimpleFigure


def showinfo(filename):
    data = np.recfromcsv(filename)['datos']
    pop = stats.RandomPopulation(data)
    print('Archivo: {}'.format(filename))
    print('La distribución es ', end='')
    if pop.is_continuous():
        print('contínua')
    else:
        print('discreta')
    freq_table = pop.frequency_distribution()
    print('Tabla de distribución de frecuencias')
    for e in freq_table:
        print('{}: {} ({:.2f} %)'.format(e.val, e.freq, e.freq * 100))
    print('La varianza es: {}'.format(pop.variance()))
    print('La esperanza es: {}'.format(pop.mean()))
    fig = SimpleFigure()
    pop.plot_histogram(fig.ax)
    fig.render()


def main():
    showinfo('datos01.csv')
    print()
    showinfo('datos02.csv')
    print()
    showinfo('datos03.csv')


if __name__ == '__main__':
    main()
