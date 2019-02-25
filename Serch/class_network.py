import numpy as np
# metodo necesario para importar la libreria desde el repositorio #
import sys
sys.path.append('math_tricks/')
from math_vect_tools import *
from operations import *
import networkx as nx


class Clique(object):

    def __init__(self, elements_residues=None):

        if elements_residues == None:
            self.elements_residues = []

