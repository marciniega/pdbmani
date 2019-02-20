#!/usr/bin/env python
# librerias que utilizaras
import numpy as np
# por si no te lee las tools o functions creadas
import sys
sys.path.append("/home/serch/pdbmani/Serch/math_tricks/")
sys.path.append("/home/serch/pdbmani/Serch/")
import math_vect_tools as mymath
# herramientas para leer pdbs
import read_pdb_tools as rpt
# calculo de distancia
from scipy.spatial.distance import pdist
# libreria de tablas
import pandas as pd
# funciones de click generadas en pandas
import funciones_CLICK as fc
# cuenta tiempo de ejecucion
import datetime
# multiprocessing
import multiprocessing
from functools import partial

# networks
import networkx as nx
# filtro distancia minima
from scipy.spatial.distance import euclidean

# Inicio de tiempo
timenow_bueno = datetime.datetime.now()

timenow = datetime.datetime.now()

# lectura de archivo
file1 = '/home/serch/pdbmani/Serch/Experimentos/pdbs/frame_1.pdb'  # sys.argv[1]
file2 = '/home/serch/pdbmani/Serch/Experimentos/pdbs/frame_2.pdb'  # sys.argv[2]

# numero de cliques, preguntar en el software para generalizarlo INPUT...
number_elements_clique = 3

# se define la estructura
pdb1 = rpt.PdbStruct(file1)
pdb2 = rpt.PdbStruct(file2)

# se lee el pdb y se agrega al objeto
pdb1.AddPdbData("%s" % file1)
pdb2.AddPdbData("%s" % file2)

pdb1.Set_SS()
pdb2.Set_SS()

# se obtienen los residuos que perteneces a la cadena de interes por default chain = 'A'
pdb11 = pdb1.GetResChain()
pdb22 = pdb2.GetResChain()

# creando tabla de estructura secundaria para filtro de SS
ss1 = fc.create_ss_table(pdb11)
ss2 = fc.create_ss_table(pdb22)

# filtro dihedral

# pdb1.SetDiheMain()
# pdb2.SetDiheMain()

pdb11[0].GetAtom('CA')

def get_df_distancias(ref):
    """Funcion para obtener el dataframe de distancias de cada residuo
    Dudas en codigo pueden revisar fc.distancia_entre_atomos en ese se basa
    esta funcion, la diferencia es que se crea con el objeto residuo"""
    # se generan listas con coordenadas y numero de atomo

    # calcula distancia y regresa dataframe
    enlaces = []
    for res1 in ref:
        for res2 in ref:
            if res2.resi >= res1.resi:
                if mymath.distance(res2.GetAtom('CA').coord, res1.GetAtom('CA').coord) < 10:
                    enlaces.append([res1.resi, res2.resi])

    # se genera la matriz de adyacencias para la red
    return enlaces


enlaces1 = (get_df_distancias(pdb11))
enlaces2 = (get_df_distancias(pdb22))

red1 = (nx.Graph(enlaces1))
red2 = (nx.Graph(enlaces2))

cliques_1, cliques_max_1 = fc.gen_cliques(red1)
cliques_2, cliques_max_1 = fc.gen_cliques(red2)


def score_ss(clq1, clq2):
    flag = 1
    for k in range(3):
        res1 = clq1[k]
        res2 = clq2[k]
        if fc.SSM(res1.ss, res2.ss) == 2:
            flag = 0
            break

    return flag


def matrix_R(vecs_c_1, vecs_c_2):

    number_of_atoms = vecs_c_1.shape[0]

    def R_ij(i, j):
        "EXPLICAR R_IJ"

        valor = sum([vecs_c_1[k, i] * vecs_c_2[k, j] for k in range(number_of_atoms)])
        return valor

    """cliques a comparar: i,j
    desde aqui se itera sobre cada i y hay que variar los vectores
    coordenada
    Regresa la matriz gigante (matriz simetrica del articulo HREF!!!!)"""

    # primer renglon
    R11R22R33 = (R_ij(0, 0) + R_ij(1, 1) + R_ij(2, 2))
    R23_R32 = (R_ij(1, 2) - R_ij(2, 1))
    R31_R13 = (R_ij(2, 0) - R_ij(0, 2))
    R12_R21 = (R_ij(0, 1) - R_ij(1, 0))
    # segundo renglon
    R11_R22_R33 = (R_ij(0, 0) - R_ij(1, 1) - R_ij(2, 2))
    R12R21 = (R_ij(0, 1) + R_ij(1, 0))
    R13R31 = (R_ij(0, 2) + R_ij(2, 0))
    # tercer renglon
    _R11R22_R33 = (-R_ij(0, 0) + R_ij(1, 1) - R_ij(2, 2))
    R23R32 = (R_ij(1, 2) + R_ij(2, 1))
    # cuarto renglon
    _R11_R22R33 = (-R_ij(0, 0) - R_ij(1, 1) + R_ij(2, 2))

    matriz_R = [
        [R11R22R33, R23_R32, R31_R13, R12_R21],
        [R23_R32, R11_R22_R33, R12R21, R13R31],
        [R31_R13, R12R21, _R11R22_R33, R23R32],
        [R12_R21, R13R31, R23R32, _R11_R22R33]
    ]
    return (np.array(matriz_R))


def rotation_matrix(matriz_R):
    """utilizando la funcion giant_matrix, fijando los valores de i,j
    se calcula la matriz de rotacion con los eigenvectores y eigenvalores
    arroja una matriz de rotacion que depende de la matriz gigante
    """
    eignvalues, eigenvectors = np.linalg.eig(matriz_R)
    q = eigenvectors[:, np.argmax(eignvalues)]

    # matriz de rotacion con eigenvectores forma USING QUATERNIONS TO CALCULATE RMSD
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    matriz_rotacion = np.array([
        [(q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), (q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)]
    ], dtype=np.float64)

    return (matriz_rotacion)


def rotation_vectors(vector_gorro, matriz_rotacion):
    """obtencion de vector rotado,
    utilizando la matriz de rotacion
    y los vectores gorro a rotar y trasladar"""
    coord_rotado = [np.matmul(matriz_rotacion, coord_atom) for coord_atom in vector_gorro]

    return (np.array(coord_rotado))


def rmsd_between_cliques(clique_trasladado_rotado, atom_to_compare):
    """Calculo de rmsd entre cliques tomando el atomo rotado y trasladado
    y el atomo a comparar"""
    # PRIMERO Y ULTIMO ESTABA MAL EL CALCULO
    # pre_rmsd = np.sum((clique_trasladado_rotado - np.array(atom_to_compare, dtype=np.float64)) ** 2, 1)
    # rmsd_i = lambda i: np.sqrt(i / 3)
    # rmsd_final = rmsd_i(pre_rmsd).mean()
    # return (rmsd_final)

    dim_coord = clique_trasladado_rotado.shape[1]
    N = clique_trasladado_rotado.shape[0]
    result = 0.0
    for v, w in zip(atom_to_compare, clique_trasladado_rotado):
        result += sum([(v[i] - w[i]) ** 2.0 for i in range(dim_coord)])

    rmsd_final = np.sqrt(result / N)

    return (rmsd_final)


def align(c_1, c_2):

    bari_1 = c_1.mean(0)
    bari_2 = c_2.mean(0)

    vecs_center_1 = c_1 - bari_1
    vecs_center_2 = c_2 - bari_2

    limite_distancia_minima = 0.45
    if number_elements_clique == 4:
        limite_distancia_minima = 0.9
    if number_elements_clique == 5:
        limite_distancia_minima = 1.8
    if number_elements_clique == 6:
        limite_distancia_minima = 3.6
    if number_elements_clique == 7:
        limite_distancia_minima = 4.5
    if number_elements_clique == 8:
        limite_distancia_minima = 8.0

    def minimum_distance():

        flag = 0
        origin = (0, 0, 0)
        dist_1 = np.mean([euclidean(origin, j) for j in vecs_center_1])
        dist_2 = np.mean([euclidean(origin, j) for j in vecs_center_2])
        if abs(dist_1 - dist_2) > limite_distancia_minima:
            flag = 1

        return flag

    if minimum_distance():
        rmsd_final = 100  # rmsd muy grande
        return rmsd_final

    matriz_R = matrix_R(vecs_center_1, vecs_center_2)
    matriz_rotacion = rotation_matrix(matriz_R)

    vector_rotado = rotation_vectors(vecs_center_1, matriz_rotacion)

    vector_rotado_trasladado_a_clique2 = vector_rotado + bari_2
    # print(vector_rotado_trasladado_a_clique2)
    protein_trasladado_rotado = vector_rotado_trasladado_a_clique2
    protein_to_compare = np.array(c_2, dtype=np.float)

    # TE PUEDES AHORRAR EL PASO DE TRASLADAR SI CALCULAS EN LOS VECTORES CENTRICOS.
    rmsd_final = rmsd_between_cliques(protein_trasladado_rotado, protein_to_compare)

    return rmsd_final


restriccion_rmsd = 0.15
if number_elements_clique == 4:
    restriccion_rmsd = 0.30
if number_elements_clique == 5:
    restriccion_rmsd = 0.60
if number_elements_clique == 6:
    restriccion_rmsd = 0.90
if number_elements_clique == 7:
    restriccion_rmsd = 1.50
if number_elements_clique == 8:
    restriccion_rmsd = 1.80


cliques_candidate = []
for clique1 in [y for x in cliques_1 for y in x]:
    res_clq_1 = [pdb1.GetRes(j) for j in clique1]
    for clique2 in [y for x in cliques_2 for y in x]:
        res_clq_2 = [pdb2.GetRes(j) for j in clique2]
        if score_ss(res_clq_1, res_clq_2):
            coord_1 = np.array([i.GetAtom('CA').coord for i in res_clq_1])
            coord_2 = np.array([i.GetAtom('CA').coord for i in res_clq_2])
            # print(align(coord_1, coord_2))
            if align(coord_1, coord_2) < restriccion_rmsd:
                print(align(coord_1, coord_2))
                new_cliques = []
                # agregando elemento
                for i in cliques_max_1:
                    if (set(clique1).issubset(i)):
                        print(clique1)
                        print("clique maximal:", i)
                        no_estan_en_clique = set(i).difference(set(clique1))
                        print(no_estan_en_clique)
                        for nuevo_elemento in no_estan_en_clique:
                            clique_nuevo = list(clique1).copy()
                            clique_nuevo = np.append(clique_nuevo, nuevo_elemento)
                            new_cliques.append(clique_nuevo)
                print(new_cliques)
                # agrega elemento

                cliques_candidate.append((clique1, clique2))

                break
    break
# print(cliques_candidate)
# print(len(cliques_candidate))

# print(cliques_candidate[0][0])


# Funcion para obtener la siguiente iteracion de candidatos.
# def add_element_clique(df_cliques, col_candidatos, cliques, candidatos_df, number_elements_clique):
#     cliques_maximales = cliques[
#         cliques.numero_elementos >= number_elements_clique].drop('numero_elementos', 1).values
#     set_candidatos = [df_cliques.iloc[i, range(
#         number_elements_clique)].values for i in candidatos_df[col_candidatos].unique()]
#     #conjunto de candidatos unicos
#     lista_residuos = []  # aqui se guardara la lista de numero de residuo
#     for candidato in set_candidatos:  # este va a cambiar cada iteracion
#         for clique_maximal in cliques_maximales:
#             clique_maximal = [i for i in clique_maximal if str(i) != 'nan']
#             if set(candidato).issubset(clique_maximal):       # si esta en un clique maximal
#                 no_estan_en_clique = set(clique_maximal).difference(set(candidato))
#                 # obten los elementos que no estan en ese clique que si estan en el clique maximal
#                 for nuevo_elemento in no_estan_en_clique:
#                     candidato_nuevo = candidato.copy()
#                     # se genera una copia para no borrar el orginal
#                     candidato_nuevo = np.append(candidato_nuevo, nuevo_elemento)
#                     # se apila un elemento de los que no estan
#                     if set(candidato_nuevo) not in lista_residuos:
#                         lista_residuos.append(set(candidato_nuevo))
#                         # si no esta en la lista pasa
#
#     return(pd.DataFrame(lista_residuos))

