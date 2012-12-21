'''
notebookutils.py
natural-mixer

2012 Brandon Mechtley
Arizona State University

A few helpful routines for using IPython notebook.
'''

from IPython.core.pylabtools import print_figure
from IPython.core.display import display, HTML, Math, Latex
from sympy import Matrix, latex

def showmat(m, prelabel='', postlabel='', eqtype='$'):
    '''Display a numpy.ndarray as a latex matrix in an IPython notebook with optional caption.
        m: np.ndarray
            array to display
        prelabel: str, optional
            Latex to insert before the matrix (default '').
        postlabel: str, optional
            Latex to insert after the matrix (default '')'''
    
    display(
        Latex(
            prelabel + eqtype + 
            latex(Matrix(m), mat_str='matrix') + 
            eqtype + postlabel
        )
    )

def svgfig(f):
    '''Display a matplotlib figure as SVG in an IPython notebook.
        f: matplotlib.figure
            figure to display as SVG
            
        Note that this routine assumes you are NOT using ipython notebook --pylab inline, as 
        otherwise the original figure will be displayed as a rasterized PNG in addition to this SVG 
        figure by default.'''
    
    display(HTML(print_figure(f, fmt='svg')))
