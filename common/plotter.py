import matplotlib.pyplot as _plt
import tikzplotlib


class Figure:

    def __init__(self):
        self._fig = _plt.figure()

    def render(self, latexfile=None, standalone_latex=False):
        for ax in self._fig.axes:
            self._axes_legend(ax)
        if latexfile is None:
            # self._fig.show()  FIXME no funciona en modo script
            _plt.show()  # FIXME esto intenta mostrar todas las figuras pendientes
        else:
            tikzcode = tikzplotlib.get_tikz_code(figure=self._fig,
                                                 extra_axis_parameters=[
                                                     'scaled ticks=false',
                                                     'xticklabel style={/pgf/number format/.cd,fixed,precision=2}',
                                                     'yticklabel style={/pgf/number format/.cd,fixed,precision=2}'
                                                 ],
                                                 standalone=standalone_latex)
            with open(latexfile, 'w') as f:
                f.write(tikzcode)

    @staticmethod
    def _axes_legend(ax):
        haslabels = False
        for line in ax.lines:
            label = line.get_label()
            if label and not label.startswith('_'):
                haslabels = True
                break
        if not haslabels:
            for patch in ax.patches:
                label = patch.get_label()
                if label and not label.startswith('_'):
                    haslabels = True
                    break
        if not haslabels:
            for coll in ax.collections:
                label = coll.get_label()
                if label and not label.startswith('_'):
                    haslabels = True
                    break
        if haslabels:
            ax.legend(fancybox=True, framealpha=0.5)


class SimpleFigure(Figure):

    def __init__(self, xlabel=None, ylabel=None):
        super().__init__()
        self.ax = self._fig.add_subplot(111, xlabel=xlabel, ylabel=ylabel)
        self.ax.grid(True)
