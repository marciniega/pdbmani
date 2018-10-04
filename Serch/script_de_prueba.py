# import numpy as np
# import numpy.linalg as np_linalg
# # metodo necesario para importar la libreria desde el repositorio #
import sys
# sys.path.append('.../math_tricks/')
# from math_vect_tools import *
# from operations import *
import read_pdb_tools as rpt


#Lectura de archivo
infile = '1tig.pdb'#sys.argv[1]
# archivo de salida
outfile = open('%s.txt' %infile.split('.')[0], 'w')

trj = rpt.Trajectory("first")
trj.ReadTraj("%s"%infile, every=1)
trj.PrintTrajInfo()
