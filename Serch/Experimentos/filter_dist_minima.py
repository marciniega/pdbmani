#!/usr/bin/env python

# librerias que utilizaras
import numpy as np
# por si no te lee las tools o functions creadas
import os
os.chdir('/home/serch/pdbmani/Serch')
import sys
sys.path.append("/home/serch/pdbmani/Serch/math_tricks/")
sys.path.append("/home/serch/pdbmani/Serch/")
import math_vect_tools as mymath
# herramientas para leer pdbs
import read_pdb_tools as rpt
# libreria de tablas
# import pandas as pd
# funciones de click generadas en pandas
import funciones_CLICK as fc
# cuenta tiempo de ejecucion
import datetime
time_all = datetime.datetime.now()
# networks
import networkx as nx
# filtro distancia minima
from scipy.spatial.distance import euclidean
# Por si no jala el script te pongas justo donde estan las cosas!!!


# multiprocessing
# import multiprocessing
# from functools import partial


# lectura de archivo
file1 = 'pdbs/1xxa_clean.pdb'  # sys.argv[1]
file2 = 'pdbs/1tig_clean.pdb'  # sys.argv[2]

# file1 = 'pdbs/2mhu.pdb'  # sys.argv[1]
# file2 = 'pdbs/2mrt.pdb'  # sys.argv[2]

# numero de cliques, preguntar en el software para generalizarlo INPUT...
number_elements_clique = 3

# se define la estructura
pdb1 = rpt.PdbStruct(file1)
pdb2 = rpt.PdbStruct(file2)

# se lee el pdb y se agrega al objeto
pdb1.AddPdbData("%s" % file1)
pdb2.AddPdbData("%s" % file2)

# filtro SS
pdb1.Set_SS()
pdb2.Set_SS()

# filtro dihedral
pdb1.SetDiheMain()
pdb2.SetDiheMain()

# se obtienen los residuos que perteneces a la cadena de interes por default chain = 'A'
pdb11 = pdb1.GetResChain()
pdb22 = pdb2.GetResChain()

# creando tabla de estructura secundaria para filtro de SS
ss1 = fc.create_ss_table(pdb11)
ss2 = fc.create_ss_table(pdb22)


def get_df_distancias(ref):
    """Funcion para obtener los enlaces de distancias de cada residuo
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

cliques_1, cliques_max_1 = fc.gen_cliques(red1, k=7)
cliques_2, cliques_max_2 = fc.gen_cliques(red2, k=7)


# FIltro score Estructura Secundaria
def score_ss(clq1, clq2):
    flag = 1
    for k in range(3):
        res1 = clq1[k]
        res2 = clq2[k]
        if fc.SSM(res1.ss, res2.ss) == 2:
            flag = 0
            break

    return flag


# Rotacion y traslacion
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


# Metodoo de quaterniones
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


# aplicacion de matriz de rotacion
def rotation_vectors(vector_gorro, matriz_rotacion):
    """obtencion de vector rotado,
    utilizando la matriz de rotacion
    y los vectores gorro a rotar y trasladar"""
    coord_rotado = [np.matmul(matriz_rotacion, coord_atom) for coord_atom in vector_gorro]

    return (np.array(coord_rotado))


# calculo de RMSD
def rmsd_between_cliques(clique_trasladado_rotado, atom_to_compare):
    """Calculo de rmsd entre cliques tomando el atomo rotado y trasladado
    y el atomo a comparar"""

    dim_coord = clique_trasladado_rotado.shape[1]
    N = clique_trasladado_rotado.shape[0]
    result = 0.0
    for v, w in zip(atom_to_compare, clique_trasladado_rotado):
        result += sum([(v[i] - w[i]) ** 2.0 for i in range(dim_coord)])

    rmsd_final = np.sqrt(result / N)

    return (rmsd_final)


# CHECK DE ALINEAMINETO
def align(c_1, c_2):

    bari_1 = c_1.mean(0)
    bari_2 = c_2.mean(0)

    vecs_center_1 = c_1 - bari_1
    vecs_center_2 = c_2 - bari_2

    def minimum_distance():

        flag = 0
        if number_elements_clique == 3:
            limite_distancia_minima = 0.13
        elif number_elements_clique == 4:
            limite_distancia_minima = 0.15
        else:
            limite_distancia_minima = 100

        origin = (0, 0, 0)
        dist_1 = np.mean([euclidean(origin, j) for j in vecs_center_1])
        dist_2 = np.mean([euclidean(origin, j) for j in vecs_center_2])
        diff_dist = abs(dist_1 - dist_2)

        if diff_dist > limite_distancia_minima:
            flag = 1

        return [diff_dist, flag]

    if minimum_distance()[1]:
        diff_dist = 100
        rmsd_final = 100
        return rmsd_final, diff_dist
    else:
        diff_dist = minimum_distance()[0]

    matriz_R = matrix_R(vecs_center_1, vecs_center_2)
    matriz_rotacion = rotation_matrix(matriz_R)

    vector_rotado = rotation_vectors(vecs_center_1, matriz_rotacion)

    vector_rotado_trasladado_a_clique2 = vector_rotado + bari_2
    # print(vector_rotado_trasladado_a_clique2)
    protein_trasladado_rotado = vector_rotado_trasladado_a_clique2
    protein_to_compare = np.array(c_2, dtype=np.float)

    # TE PUEDES AHORRAR EL PASO DE TRASLADAR SI CALCULAS EN LOS VECTORES CENTRICOS.
    rmsd_final = rmsd_between_cliques(protein_trasladado_rotado, protein_to_compare)

    return rmsd_final, diff_dist


cliques_1_temp = [y for x in cliques_1 for y in x]
cliques_2_temp = [y for x in cliques_2 for y in x]


print("numero de cliques1", len(cliques_1_temp))
print("numero de cliques2", len(cliques_2_temp))
print("numero de candidatos %1.2f" % (len(cliques_1_temp) * len(cliques_2_temp)))


def add_element_to_clique(cliques_candidatos,
                          cliques_maximales,
                          proteina):
    """
    :param cliques_candidatos: lista de cliques canidatos
    :param cliques_maximales: lista de cliques maximales
    :param proteina: 0 o 1
    :return: Lista de cliques nuevos
    """
    cliques_nuevos = []
    # agregando elemento
    for clique in cliques_candidatos:
        for clique_max in cliques_maximales:
            if set(clique[proteina]).issubset(clique_max):
                # print(clique1[0])
                # print("clique maximal:", i)
                no_estan_en_clique = set(clique_max).difference(set(clique[proteina]))
                # print(no_estan_en_clique)
                for nuevo_elemento in no_estan_en_clique:
                    clique_nuevo = list(clique[proteina]).copy()
                    clique_nuevo = np.append(clique_nuevo, nuevo_elemento)
                    if clique_nuevo.tolist() not in cliques_nuevos:
                        cliques_nuevos.append(clique_nuevo.tolist())

    return cliques_nuevos


def iter_align(number_elements_clique, cliques_1_align, cliques_2_align):

    print("Numero de cliques nuevos proteina 1", len(cliques_1_align))
    print("Numero de cliques nuevos proteina 2", len(cliques_2_align))
    print("numero de candidatos %1.5f" % (len(cliques_1_align) * len(cliques_2_align)))
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
    for clique1 in cliques_1_align:
        # if len(set(tuple(clique1)).intersection([6, 24, 23])) > 1:
            res_clq_1 = [pdb1.GetRes(clq) for clq in clique1]
            for clique2 in cliques_2_align:
                res_clq_2 = [pdb2.GetRes(clq) for clq in clique2]
                if score_ss(res_clq_1, res_clq_2):
                    coord_1 = np.array([res.GetAtom('CA').coord for res in res_clq_1])
                    coord_2 = np.array([res.GetAtom('CA').coord for res in res_clq_2])
                    # if align(coord_1, coord_2) < restriccion_rmsd:
                    # print("RMSD: %1.5f" % align(coord_1, coord_2))
                    # print(clique1, clique2)

                    cliques_candidate.append((clique1, clique2, align(coord_1, coord_2)))

    if number_elements_clique < 7:
        new_cliques_1 = add_element_to_clique(cliques_candidate,
                                              cliques_max_1,
                                              0)

        new_cliques_1 = tuple(tuple(x) for x in random.sample(new_cliques_1, 1000))
        new_cliques_2 = add_element_to_clique(cliques_candidate,
                                              cliques_max_2,
                                              1)
        new_cliques_2 = tuple(tuple(x) for x in random.sample(new_cliques_2, 1000))

        number_elements_clique = number_elements_clique + 1


    else:
        new_cliques_1 = cliques_1_align
        new_cliques_2 = cliques_2_align
        number_elements_clique = number_elements_clique + 1

    return number_elements_clique, new_cliques_1, new_cliques_2, cliques_candidate


import random
random.seed(568)
new_df_cliques1 = random.sample(cliques_1_temp, 1000)
new_df_cliques2 = random.sample(cliques_2_temp, 1000)
import pandas as pd
for i in range(5):
    number_elements_clique, new_df_cliques1, new_df_cliques2, cliques_candidatos, = iter_align(
        number_elements_clique, new_df_cliques1, new_df_cliques2)
    pd.DataFrame(cliques_candidatos).to_pickle('cliques_rmsd_dist'+str(number_elements_clique - 1)+'.pkl')
    print("iteracion", i + 1, "numero de elementos:", number_elements_clique)
    print("===" * 10)


timenow = datetime.datetime.now()
print('Tiempo Total:', timenow - time_all)
