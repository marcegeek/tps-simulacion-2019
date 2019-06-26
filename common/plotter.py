import matplotlib.pyplot as plt
import matplotlib2tikz


def render(latexfile=None, standalone_latex=False):
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
