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

import networkx as nx

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

cliques_max_1 = fc.gen_cliques(red1)
cliques_max_2 = fc.gen_cliques(red2)

# print(cliques_max_2)
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

    def R_ij(i, j):
        "EXPLICAR R_IJ"
        number_of_atoms = vecs_c_1.shape[0]
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
    return (matriz_R)

def rotation_matrix(matriz_R):
    """utilizando la funcion giant_matrix, fijando los valores de i,j
    se calcula la matriz de rotacion con los eigenvectores y eigenvalores
    arroja una matriz de rotacion que depende de la matriz gigante
    """
    eignvalues, eigenvectors = np.linalg.eig(matriz_R)
    q = eigenvectors[:, np.argmax(eignvalues)]
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    # matriz de rotacion con eigenvectores
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

    coord_rotado = [np.matmul(
        matriz_rotacion, i.reshape(3, 1)).T[0] for i in vector_gorro]
    return (coord_rotado)


def rmsd_between_cliques(clique_trasladado_rotado, atom_to_compare):
    """Calculo de rmsd entre cliques tomando el atomo rotado y trasladado
    y el atomo a comparar, por el momento solo imprime el resultado"""
    # primer RMSD entre atomos
    pre_rmsd = np.sum((np.array(
        atom_to_compare, dtype=np.float64) - clique_trasladado_rotado) ** 2, 1)
    rmsd_i = lambda i: np.sqrt(i) / 3
    rmsd_final = rmsd_i(pre_rmsd).mean()

    return (rmsd_final)


def align(c_1, c_2):

    bari_1 = np.array(c_1).mean(0)
    bari_2 = np.array(c_2).mean(0)

    vecs_center_1 = c_1 - bari_1
    vecs_center_2 = c_2 - bari_2

    matriz_R = matrix_R(vecs_center_1, vecs_center_2)
    matriz_rotacion = rotation_matrix(matriz_R)
    vector_rotado = rotation_vectors(vecs_center_1, matriz_rotacion)
    vector_rotado_trasladado_a_clique2 = vector_rotado + bari_2
    protein_trasladado_rotado = vector_rotado_trasladado_a_clique2
    protein_to_compare = np.array(c_2, dtype=np.float)
    print(protein_trasladado_rotado, protein_to_compare)
    print('resta')
    print(np.array(
        protein_to_compare, dtype=np.float64) - protein_trasladado_rotado)
    print('cuadrado')
    print((np.array(
        protein_to_compare, dtype=np.float64) - protein_trasladado_rotado) ** 2)
    print('suma')
    print(np.sum((np.array(
          protein_to_compare, dtype=np.float64) - protein_trasladado_rotado) ** 2, 1))

    pre_rmsd = np.sum((np.array(
        protein_to_compare, dtype=np.float64) - protein_trasladado_rotado) ** 2, 1)
    print(pre_rmsd)
    rmsd_i = lambda i: np.sqrt(i) / 3
    print('rmsd_indi')
    print(rmsd_i(pre_rmsd))
    rmsd_final = rmsd_i(pre_rmsd).mean()

    rmsd_final = rmsd_between_cliques(protein_trasladado_rotado, protein_to_compare)

    return rmsd_final


cliques_candidate = []
for clique1 in [y for x in cliques_max_1 for y in x]:
    res_clq_1 = [pdb1.GetRes(j) for j in clique1]
    for clique2 in [y for x in cliques_max_2 for y in x]:
        res_clq_2 = [pdb2.GetRes(j) for j in clique2]
        if score_ss(res_clq_1, res_clq_2):
            coord_1 = [i.GetAtom('CA').coord for i in res_clq_1]
            coord_2 = [i.GetAtom('CA').coord for i in res_clq_2]
            # print(coord_1, coord_2)
            print(align(coord_1, coord_2))
            print([i.resi for i in res_clq_1])
            print([i.resi for i in res_clq_2])
            cliques_candidate.append((clique1,clique2))
        break
    break

# print(cliques_candidate)

# def iter_rmsd(new_df_cliques1, new_df_cliques2, number_elements_clique):
#     if number_elements_clique == 7:
#         print("Filtrando candidatos y preparando datos para alineamiento")
#
#     # filtro residuos de 7 unicos
#     for col in new_df_cliques1.columns:
#         mask = np.where(new_df_cliques1[col].isin(residuos_unicos_1), True, False)
#         new_df_cliques1 = new_df_cliques1[mask].reset_index(drop=True)
#
#     for col in new_df_cliques2.columns:
#         mask = np.where(new_df_cliques2[col].isin(residuos_unicos_2), True, False)
#         new_df_cliques2 = new_df_cliques2[mask].reset_index(drop=True)
#
#     # filtro estructura secundaria
#     new_df_cliques1 = fc.paste_SS(ss1, new_df_cliques1, num_cliques=number_elements_clique)
#     new_df_cliques2 = fc.paste_SS(ss2, new_df_cliques2, num_cliques=number_elements_clique)
#     candidatos_ss = fc.compare_SS(new_df_cliques1, new_df_cliques2, num_cliques=number_elements_clique)
#
#     # rotando y trasladando
#     df_cliques1 = fc.get_coords_clique(df_atoms1, new_df_cliques1, number_elements_clique)
#     df_cliques2 = fc.get_coords_clique(df_atoms2, new_df_cliques2, number_elements_clique)
#     df_cliques1 = fc.baricenter_clique(df_cliques1, number_elements_clique)
#     df_cliques2 = fc.baricenter_clique(df_cliques2, number_elements_clique)
#     df_cliques1 = fc.center_vectors(df_cliques1, number_elements_clique)
#     df_cliques2 = fc.center_vectors(df_cliques2, number_elements_clique)
#     idx_rmsd1, idx_rmsd2 = 3 * number_elements_clique, 4 * number_elements_clique + 3  # guardas la posicion de los vectores
#     array_df_cliques1 = df_cliques1.values[:, range(idx_rmsd1, idx_rmsd2)]  # del 9 al 15
#     array_df_cliques2 = df_cliques2.values[:, range(idx_rmsd1, idx_rmsd2)]
#
#     # filtro de distancia minima
#     df_cliques1 = fc.get_distancia_promedio(number_elements_clique, df_cliques1)
#     df_cliques2 = fc.get_distancia_promedio(number_elements_clique, df_cliques2)
#     array_dist_promedio1 = df_cliques1.values[:, -1]  # el ultimo valor de distancia.
#     array_dist_promedio2 = df_cliques2.values[:, -1]
#     limite_distancia_minima = 0.45
#     if number_elements_clique == 4:
#         limite_distancia_minima = 0.9
#     if number_elements_clique == 5:
#         limite_distancia_minima = 1.8
#     if number_elements_clique == 6:
#         limite_distancia_minima = 3.6
#     if number_elements_clique == 7:
#         limite_distancia_minima = 4.5
#     if number_elements_clique == 8:
#         limite_distancia_minima = 8.0
#
#     # filtro por distancia minima
#     candidatos_filter_dist = [(candidato_1, candidato_2) for candidato_1, candidato_2 in candidatos_ss if (
#             array_dist_promedio1[candidato_1] - array_dist_promedio2[candidato_2] >= -limite_distancia_minima) & (
#                                       array_dist_promedio1[candidato_1] - array_dist_promedio2[
#                                   candidato_2] <= limite_distancia_minima)]
#
#     # filtro por restriccion de RMSD despues de ajuste 3D
#     p = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
#     restriccion_rmsd = 0.15
#     if number_elements_clique == 4:
#         restriccion_rmsd = 0.30
#     if number_elements_clique == 5:
#         restriccion_rmsd = 0.60
#     if number_elements_clique == 6:
#         restriccion_rmsd = 0.90
#     if number_elements_clique == 7:
#         restriccion_rmsd = 1.50
#     if number_elements_clique == 8:
#         restriccion_rmsd = 1.80
#
#     rmsd_1 = p.map(partial(fc.calculate_rmsd_rot_trans_m,
#                            array_cliques1=array_df_cliques1,
#                            array_cliques2=array_df_cliques2,
#                            num_cliques=number_elements_clique), candidatos_filter_dist)
#
#     p.close()
#     p.join()
#
#     df_candidatos = pd.DataFrame(rmsd_1, columns=['rmsd', 'candidatos', 'matriz_rotacion'])
#     df_candidatos = df_candidatos[df_candidatos.rmsd <= restriccion_rmsd]
#     df_candidatos['candidato_clique_1'] = df_candidatos.candidatos.str.get(0)
#     df_candidatos['candidato_clique_2'] = df_candidatos.candidatos.str.get(1)
#     time = datetime.datetime.now()
#
#     print('numero de candidatos:', df_candidatos.shape[0])
#     print('tiempo pasado:', time - timenow)
#
#     # Se agrega un nuevo elemento a los cliques.
#     if number_elements_clique < 7:
#         new_df_cliques1 = fc.add_element_clique(df_cliques1, 'candidato_clique_1', cliques1, df_candidatos,
#                                                 number_elements_clique)
#         new_df_cliques2 = fc.add_element_clique(df_cliques2, 'candidato_clique_2', cliques2, df_candidatos,
#                                                 number_elements_clique)
#         number_elements_clique = number_elements_clique + 1
#
#     return (new_df_cliques1, new_df_cliques2, number_elements_clique, df_candidatos)
#
#
# # primeros cliques a filtrar
# new_df_cliques1 = df_cliques1
# new_df_cliques2 = df_cliques2
#
# # Si se empieza en 3-cliques solo itera 5 veces para llegar a 7 :v.
# for k in range(1):
#     new_df_cliques1, new_df_cliques2, number_elements_clique, rmsd = iter_rmsd(
#         new_df_cliques1, new_df_cliques2, number_elements_clique)
#     print("iteracion", k + 1, "numero de elementos:", number_elements_clique)
#     print("===" * 10)
#
#
# print(rmsd)
# print('//')
# print(new_df_cliques1[1])
# print('//')
# print(new_df_cliques2[1])