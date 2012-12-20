from IPython.core.pylabtools import print_figure
from IPython.core.display import display, HTML, Math, Latex
from sympy import Matrix, latex

def showmat(m, prelabel='', postlabel=''):
    '''Display a numpy.ndarray as a latex matrix in an IPython notebook with optional caption.
        m: np.ndarray
            array to display
        prelabel: str, optional
            Latex to insert before the matrix (default '').
        postlabel: str, optional
            Latex to insert after the matrix (default '')'''
    
    display(
        Latex(
            prelabel + '$' + 
            latex(Matrix(m), mat_str='matrix') + 
            '$' + postlabel
        )
    )

def svgfig(f):
    '''Display a matplotlib figure as SVG in an IPython notebook.
        f: matplotlib.figure
            figure to display as SVG'''
    
    display(HTML(print_figure(f, fmt='svg')))
