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
# funciones de click generadas en pandas
import funciones_CLICK as fc
# cuenta tiempo de ejecucion
import datetime
time_all = datetime.datetime.now()
# networks
import networkx as nx
# # filtro distancia minima
# from scipy.spatial.distance import euclidean
# por si no jala
import os
os.chdir('/home/serch/pdbmani/Serch')
# multiprocessing
# import multiprocessing
# from functools import partial

# lectura de archivo
file1 = '/home/serch/pdbmani/Serch/pdbs/1xxa_clean.pdb'  # sys.argv[1]
file2 = '/home/serch/pdbmani/Serch/pdbs/1tig_clean.pdb'  # sys.argv[2]
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

#filtro SS
pdb1.Set_SS()
pdb2.Set_SS()

# filtro dihedral
pdb1.SetDiheMain()
pdb2.SetDiheMain()

# se obtienen los residuos que perteneces a la cadena de interes por default chain = 'A'
pdb11 = pdb1.GetResChain()
pdb22 = pdb2.GetResChain()

if len(pdb22) < len(pdb11):

    import copy
    pdb1_temp = copy.copy(pdb1)
    pdb2_temp = copy.copy(pdb2)

    pdb11_temp = copy.copy(pdb11)
    pdb22_temp = copy.copy(pdb22)

    pdb1 = pdb2_temp
    pdb2 = pdb1_temp

    pdb11 = pdb22_temp
    pdb22 = pdb11_temp

    del [pdb1_temp]
    del [pdb2_temp]
    del [pdb11_temp]
    del [pdb22_temp]

    print("Intercambio de nombre ya que la proteina 1 es mas grande que la 2")
    print(pdb1.name, len(pdb11))
    print(pdb2.name, len(pdb22))
    print("No te preocupes ya quedo :)")

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
    for res1 in ref[1:-1]:  # ref[1:30]
        for res2 in ref[1:-1]:  # ref[1:40]
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

lenght_cliquemax_1 = len(cliques_max_1)
lenght_cliquemax_2 = len(cliques_max_2)

print("numero de cliques maximales combinaciones", lenght_cliquemax_1 * lenght_cliquemax_2)


def eval_dihedral(ang_ref, ang_tar, cutoff=30):
    """
    Evaluacion de los angulo dihedrales, manteniendo aquellos que presentan un cierto cutoff.
    :param ang_ref: angulo phi o psi
    :param ang_tar: angulo phi o psi
    :return: flag (1 o 0)
    """
    if ang_ref * ang_tar > 0:
        if abs(ang_ref - ang_tar) < cutoff:
            return 1
    elif ang_ref < 0:
        ang_ref = 360 - ang_ref
    elif ang_tar < 0:
        ang_tar = 360 - ang_tar

    if abs(ang_ref - ang_tar) < cutoff:
            return 1

    return 0

###############
# Alineamiento#
###############
list_candidates = []
for clique1 in cliques_max_1:
    res_clq_1 = [pdb1.GetRes(j) for j in clique1]
    for clique2 in cliques_max_2:
        res_clq_2 = [pdb2.GetRes(j) for j in clique2]

        # Filtro PHI PSI
        val_vec = []
        for res1 in res_clq_1:
            phi_ref = res1.phi
            psi_ref = res1.psi
            val = 0
            for res2 in res_clq_2:
                phi_tar = res2.phi
                psi_tar = res2.psi
                if eval_dihedral(phi_ref, phi_tar, cutoff=8) and (
                        eval_dihedral(psi_ref, psi_tar, cutoff=8)):
                    val = val + 1
            val_vec.append(val)
        if val_vec.count(0) < 1:
            # debugg de vector de angulos dihedrales
            list_candidates.append([clique1, clique2])


# # numero de candidatos y parejas generadas.
print("numero de candidatos despues de filtro dihedral", len(list_candidates))
print(list_candidates)
# Filtro score Estructura Secundaria
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
        """
        Funcion que calcula el valor de R_ij basado en Using quaternions to calculate RMSD
        :float i  Coordenada (0=x,y=1,z=2):
        :float j Coordenada (0=x,y=1,z=2):
        :return : valor R_ij
        """
        valor = sum([vecs_c_1[k, i] * vecs_c_2[k, j] for k in range(number_of_atoms)])
        return valor

    """cliques a comparar: i,j
    desde aqui se itera sobre cada i y hay que variar los vectores
    coordenada
    Regresa la matriz gigante 
    (matriz simetrica del articulo Using quaternions to calculate RMSD!!!!)"""

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
    coord_rotado = np.array([np.matmul(matriz_rotacion, coord_atom) for coord_atom in vector_gorro])

    return coord_rotado


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
def align(c_1, c_2, number_elements_clique = None):

    bari_1 = c_1.mean(0)
    bari_2 = c_2.mean(0)

    vecs_center_1 = c_1 - bari_1
    vecs_center_2 = c_2 - bari_2

    # if number_elements_clique == 3:
    #     limite_distancia_minima = 0.13
    # elif number_elements_clique == 4:
    #     limite_distancia_minima = 0.15
    # elif number_elements_clique == 5:
    #     limite_distancia_minima = 0.60
    # elif number_elements_clique == 6:
    #     limite_distancia_minima = 0.60
    # elif number_elements_clique == 7:
    #     limite_distancia_minima = 0.60
    # else:
    #     limite_distancia_minima = 100
    #
    # def minimum_distance():
    #
    #     flag = 0
    #     origin = (0, 0, 0)
    #     dist_1 = np.mean([euclidean(origin, j) for j in vecs_center_1])
    #     dist_2 = np.mean([euclidean(origin, j) for j in vecs_center_2])
    #     if abs(dist_1 - dist_2) > limite_distancia_minima:
    #         flag = 1
    #
    #     return flag
    #
    # if minimum_distance():
    #     rmsd_final = 100  # rmsd muy grande
    #     return rmsd_final

    matriz_R = matrix_R(vecs_center_1, vecs_center_2)
    matriz_rotacion = rotation_matrix(matriz_R)

    vector_rotado = rotation_vectors(vecs_center_1, matriz_rotacion)

    protein_trasladado_rotado = vector_rotado + bari_2

    protein_to_compare = c_2

    # TE PUEDES AHORRAR EL PASO DE TRASLADAR SI CALCULAS EN LOS VECTORES CENTRICOS.
    rmsd_final = rmsd_between_cliques(protein_trasladado_rotado, protein_to_compare)

    if number_elements_clique == 7:
        return rmsd_final, matriz_rotacion

    return rmsd_final


# GENERACION DE CLIQUES DE LA LISTA DE CLIQUES MAXIMALES APLICANDO FILTRO DIHEDRAL
cliques_1_temp = []
for clique1 in list_candidates:
    list_candidate = fc.gen_cliques_3(clique1[0])
    if list_candidate not in cliques_1_temp:
        cliques_1_temp.append(list_candidate)
        # print(cliques_1_temp)

cliques_2_temp = []
for clique2 in list_candidates:
    list_candidate = fc.gen_cliques_3(clique2[1])
    if list_candidate not in cliques_2_temp:
        cliques_2_temp.append(list_candidate)
        # print(cliques_2_temp)


def iter_align(number_elements_clique, cliques_1_align, cliques_2_align):

    """
    :int number_elements_clique: elementos del clique
    :tuple cliques_1_align: lista de cliques a alinear de la proteina A
    :tuple cliques_2_align: lista de cliques a alinear de la proteina B
    :return: cliques_candidate lista de cliques candidatos
    """

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

    if number_elements_clique == 3:
        for pareja in zip(cliques_1_align, cliques_2_align):
            for clique1 in pareja[0]:
                res_clq_1 = [pdb1.GetRes(clq) for clq in clique1]
                for clique2 in pareja[1]:
                    res_clq_2 = [pdb2.GetRes(clq) for clq in clique2]
                    if score_ss(res_clq_1, res_clq_2):
                        coord_1 = np.array([res.GetAtom('CA').coord for res in res_clq_1])
                        coord_2 = np.array([res.GetAtom('CA').coord for res in res_clq_2])
                        if align(coord_1, coord_2) < restriccion_rmsd:
                            cliques_candidate.append([clique1, clique2])

    else:
        for clique1 in cliques_1_align:
            res_clq_1 = [pdb1.GetRes(clq) for clq in clique1]
            for clique2 in cliques_2_align:
                res_clq_2 = [pdb2.GetRes(clq) for clq in clique2]
                if score_ss(res_clq_1, res_clq_2):
                    coord_1 = np.array([res.GetAtom('CA').coord for res in res_clq_1])
                    coord_2 = np.array([res.GetAtom('CA').coord for res in res_clq_2])
                    if number_elements_clique == 7:
                        rmsd, mat_rot = align(coord_1, coord_2, number_elements_clique=number_elements_clique)
                        if rmsd < restriccion_rmsd:
                            cliques_candidate.append([clique1, clique2, mat_rot])

                    else:
                        if align(coord_1, coord_2) < restriccion_rmsd:
                            cliques_candidate.append([clique1, clique2])

    return cliques_candidate


print('================ alineamiento de 3-clique y agrego el 4 elemento ===================')

new_df_cliques1 = cliques_1_temp
new_df_cliques2 = cliques_2_temp

cliques_candidatos = iter_align(number_elements_clique, new_df_cliques1, new_df_cliques2)

new_df_cliques = fc.add_element_to_clique(cliques_candidatos, cliques_max_1, cliques_max_2)

number_elements_clique = number_elements_clique + 1


# print(cliques_candidatos[2])
# print(new_df_cliques[2])

print("candidatos con n-cliques, n =", number_elements_clique-1,
      "numero de parejas", len(cliques_candidatos))

print(number_elements_clique, len(new_df_cliques))

print('==================ya acabo con 3-cliques va con 4=========================')
cliques_temp = []
cliques_temp_add = []

for parejas_4clique in new_df_cliques:

    new_df_cliques1 = parejas_4clique[0]
    new_df_cliques2 = parejas_4clique[1]

    cliques_candidatos = iter_align(number_elements_clique, new_df_cliques1, new_df_cliques2)

    if cliques_candidatos != []:
        cliques_temp.append(cliques_candidatos)
        cliques_temp_add.append(fc.add_element_to_clique(cliques_candidatos, cliques_max_1, cliques_max_2))

number_elements_clique = number_elements_clique + 1
print(cliques_temp[:5])
print(len(cliques_temp))

print(cliques_temp_add[:5])
print(len(cliques_temp_add))

print('==================ya acabo con 4-cliques va con 5=========================')

cliques_temp_add_0 = [y for x in cliques_temp_add for y in x]

cliques_temp1 = []
cliques_temp1_add = []
for parejas_5clique in cliques_temp_add_0:

    new_df_cliques1 = parejas_5clique[0]
    new_df_cliques2 = parejas_5clique[1]
    cliques_candidatos = iter_align(number_elements_clique, new_df_cliques1, new_df_cliques2)

    if cliques_candidatos != []:
        cliques_temp1.append(cliques_candidatos)
        cliques_temp1_add.append(fc.add_element_to_clique(cliques_candidatos, cliques_max_1, cliques_max_2))

number_elements_clique = number_elements_clique + 1

print(cliques_temp1[:5])
print(len(cliques_temp1))

print(cliques_temp1_add[:5])
print(len(cliques_temp1_add))
print('==================ya acabo con 5-cliques va con 6=========================')

cliques_temp_add_11 = [y for x in cliques_temp1_add for y in x]

cliques_temp2 = []
cliques_temp2_add = []
for parejas_6clique in cliques_temp_add_11:

    new_df_cliques1 = parejas_6clique[0]
    new_df_cliques2 = parejas_6clique[1]
    cliques_candidatos = iter_align(number_elements_clique, new_df_cliques1, new_df_cliques2)

    if cliques_candidatos != []:
        cliques_temp2.append(cliques_candidatos)
        cliques_temp2_add.append(fc.add_element_to_clique(cliques_candidatos, cliques_max_1, cliques_max_2))

number_elements_clique = number_elements_clique + 1

print(cliques_temp2[:5])
print(len(cliques_temp2))

print(cliques_temp2_add[:5])
print(len(cliques_temp2_add))
print('==================ya acabo con 6-cliques va con 7=========================')

cliques_temp_add_22 = [y for x in cliques_temp2_add for y in x]

cliques_temp3 = []

for parejas_7clique in cliques_temp_add_22:

    new_df_cliques1 = parejas_7clique[0]
    new_df_cliques2 = parejas_7clique[1]
    cliques_candidatos = iter_align(number_elements_clique, new_df_cliques1, new_df_cliques2)

    if cliques_candidatos != []:
        cliques_temp3.append(cliques_candidatos)

print(len(cliques_temp3))
print([[x[0][0], x[0][1]] for x in cliques_temp3[:5]])
print('==================se imprimen candidatos alineables=========================')
# APARTIR DE AQUI VA EL ALINEAMIENTO DE RESIDUOS POR MEDIO DE LAS PAREJAS CANDIDATAS
parejas_cliques_finales = cliques_temp3


print(parejas_cliques_finales[0])
print(len(parejas_cliques_finales))

import pandas as pd
pd.to_pickle(pd.DataFrame(parejas_cliques_finales), 'parejas_alineables.pkl')

timenow = datetime.datetime.now()
print('Tiempo Total:', timenow - time_all)
