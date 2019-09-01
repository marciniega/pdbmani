# coding: utf-8

# libreria de analisis de datos y una caracterizacion para su facil lectura.
import pandas as pd
# libreria de generacion de rede y cliques
import networkx as nx, community

# libreria de calculo de distancia euclidiana
from scipy.spatial.distance import pdist, euclidean

# libreria de mate
import numpy as np

# libreria de iteraciones
import itertools as it

# Libreria de MA para RMSD
import sys
sys.path.append('math_tricks/')

import math_vect_tools as mymath

# verificar orden de pdbs1
import copy
# align res-res
import math


def verify_file_order(pdb1, pdb2, pdb11, pdb22):
    """
    Verifica que archivo es mas grande y lo cambia para que siempre alinea
    el pdb mas pequenio al otro mas grande.
    :param pdb1: object PdbStruct
    :param pdb2: object PdbStruct
    :param pdb11: list of residues
    :param pdb22: list of residues
    :return: pdb's and lists
    """
    if len(pdb22) < len(pdb11):

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
        print("No te preocupes ya quedo :) pero te recomendamos que cambies el orden de los files")

    return pdb1, pdb2, pdb11, pdb22


def get_df_distancias(ref):
    """Funcion para obtener los enlaces de distancias de cada residuo
    Dudas en codigo pueden revisar fc.distancia_entre_atomos en ese se basa
    esta funcion, la diferencia es que se crea con el objeto residuo"""
    # se generan listas con coordenadas y numero de atomo

    # calcula distancia y regresa dataframe
    enlaces = []
    for res1 in ref[1:-1]:
        for res2 in ref[1:-1]:
            if res2.resx >= res1.resx:
                if mymath.distance(res2.GetAtom('CA').coord, res1.GetAtom('CA').coord) < 10:
                    enlaces.append([res1.resx, res2.resx])

    # se genera la matriz de adyacencias para la red
    return enlaces


def eval_dihedral(ang_ref, ang_tar, cutoff=30):
    """
    Evaluacion de los angulo dihedrales, manteniendo aquellos que presentan un cierto cutoff.
    :param ang_ref: angulo phi o psi
    :param ang_tar: angulo phi o psi
    :param cutoff: valor de corte para filtro
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


# Filtro score Estructura Secundaria
def score_ss(clq1, clq2):
    """

    :param clq1: pareja clique de proteina A
    :param clq2: pareja clique de proteina B
    :return: flag [0,1] si cumple o no.
    """

    flag = 1
    for k in range(3):
        res1 = clq1[k]
        res2 = clq2[k]
        if SSM(res1.ss, res2.ss) == 2:
            flag = 0
            break

    return flag


# Rotacion y traslacion
def matrix_R(vecs_c_1, vecs_c_2):
    """

    :param vecs_c_1: coordenadas del vector del residuo y clique
    :param vecs_c_2: coordenadas del vector del residuo y clique
    :return: matriz de rotacion con las parejas de cliques seleccionados
    """

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


def gen_cliques(red, k=7):
    """

    :param red: Graph de networkx
    :param k: numero de elementos en el clique
    :return: lista de cliques unicos
    """

    cliques_completos = [clq for clq in nx.find_cliques(red) if len(clq) >= k]  # mayor o igual que cambiarlo
    print('numero de cliques maximos encontrados:', len(cliques_completos))

    lista_cliques = []
    for v in (cliques_completos):
        a = list(it.combinations(v, 3))
        for j in a:
            permutation_temp = it.permutations(j)
            lista_cliques.append(set(permutation_temp))

    lista_cliques = np.unique(lista_cliques)

    return lista_cliques, cliques_completos


def gen_cliques_3(red):
    """

    :param red: enlaces
    :return: lista de 3-cliques
    """

    combinations = list(it.combinations(red, 3))
    lista_cliques = [list(it.permutations(nclique)) for nclique in combinations]
    lista_cliques1 = [list(clq) for combinations in lista_cliques for clq in combinations]

    return lista_cliques1


def create_ss_table(list_residues):
    """

    :param list_residues:
    :return: DataFrame con valores de estructura secundaria, numero de residuo y cadena
    """
    ss_list = [i.ss for i in list_residues]
    num_resi_list = [i.resx for i in list_residues]
    chain_list = [i.chain for i in list_residues]

    ss = pd.DataFrame()
    ss['structure'] = ss_list
    ss['residue_number'] = num_resi_list
    ss['chain'] = chain_list
    return ss


def add_element_to_clique(cliques_candidatos,
                          cliques_maximales_1,
                          cliques_maximales_2):
    """
    Agrega un residuo del clique maximal si el clique completo es un subconjunto del clique maximal
    :param cliques_candidatos: lista de cliques canidatos
    :param cliques_maximales_1: lista de cliques maximales proteina A
    :param cliques_maximales_2: lista de cliques maximales proteina B
    :return: Lista de cliques nuevos
    """
    cliques_nuevos = []
    # agregando elemento
    for clique in cliques_candidatos:
        cliques1_nuevos = []
        cliques2_nuevos = []
        for i in cliques_maximales_1:

            if set(clique[0]).issubset(i):
                no_estan_en_clique = set(i).difference(set(clique[0]))
                for nuevo_elemento in no_estan_en_clique:
                    clique_nuevo1 = list(clique[0]).copy()
                    clique_nuevo1 = np.append(clique_nuevo1, nuevo_elemento)
                    if clique_nuevo1.tolist() not in cliques1_nuevos and len(clique_nuevo1) > 1:
                        cliques1_nuevos.append(clique_nuevo1.tolist())

        for j in cliques_maximales_2:
            if set(clique[1]).issubset(j):
                no_estan_en_clique = set(j).difference(set(clique[1]))
                for nuevo_elemento in no_estan_en_clique:
                    clique_nuevo2 = list(clique[1]).copy()
                    clique_nuevo2 = np.append(clique_nuevo2, nuevo_elemento)
                    if clique_nuevo2.tolist() not in cliques2_nuevos and len(clique_nuevo2) > 1:
                        cliques2_nuevos.append(clique_nuevo2.tolist())

        if cliques1_nuevos == [] or cliques2_nuevos == []:
            continue
        else:
            cliques_nuevos.append([cliques1_nuevos, cliques2_nuevos])

    return cliques_nuevos


def SSM(ss1, ss2):

    """Catalogo SSM siguiendo la tabla 1 y con una funcion extra,
    ss1: string (H,B,C)
    ss2: string (H,B,C)
    devuelve el score: int (0,1,2)"""
    def get_score_from_table(ss1, ss2):

        if (ss1 == 'H') and (ss2 == 'B'):
            score_ss = 2
        elif (ss1 == 'B') and (ss2 == 'H'):
            score_ss = 2
        else:
            print(ss1.ss2)
            print('WTF are u doing!')

        return(score_ss)

    score_ss = 123

    if ss1 == ss2:
        score_ss = 0
    elif (ss1 != ss2) & ((ss1 == 'C') | (ss2 == 'C')):
        score_ss = 1
    else:
        score_ss = get_score_from_table(ss1, ss2)
        
    return(score_ss)


def fun_resiudos_match(protein_trasladado_rotado, protein_to_compare, res_1, res_2):
    """
    genera los Match de parejas de residuos que cumplen un treshold de distancia (RMSD)
    :param protein_trasladado_rotado:
    :param protein_to_compare:
    :param res_1:
    :param res_2:
    :return: Lista ordenada de parejas de residuos y su distancia
    """
    return sorted([[math.sqrt(sum((c_2 - c_1) ** 2)), (res1.resx, res2.resx)] for c_1, res1 in zip(
        protein_trasladado_rotado, res_1) for c_2, res2 in zip(
        protein_to_compare, res_2) if math.sqrt(sum((c_2 - c_1) ** 2)) < 3.5])


def filter_pairs(residuos_match, flag=None):
    """
    Filtra las parejas de residuos de dos maneras si flag = False or None
    solo toma parejas que no esten repetidas, si flag = True toma todas las parejas
    y solo por orden las filtra
    :param residuos_match:
    :param flag:
    :return: distancia y parejas seleccionadas.
    """
    if flag:
        pairs_1 = [c[1][0] for c in residuos_match]
        repeat_1 = [i for i in pairs_1 if pairs_1.count(i) > 2]

        pairs_2 = [c[1][1] for c in residuos_match]
        repeat_2 = [i for i in pairs_2 if pairs_2.count(i) > 2]
    else:
        repeat_1, repeat_2 = [], []

    c1 = []
    c2 = []
    cand_n = []
    for i in residuos_match:
        if (i[1][0] in c1) or (i[1][1] in c2) or (i[1][0] in repeat_1) or (i[1][1] in repeat_2):
            continue
        else:
            c1.append(i[1][0])
            c2.append(i[1][1])
            cand_n.append(i)

    return(cand_n)


def filter_candidates(pc, flag=True):
    """

    :param pc: lista de candidatos parejas de cliques
    :param flag: si es la primera o la ultima iteracion
    :return: lista reducida de parejas candidatas
    """
    if flag:
        pc = np.array([i[0] for i in pc])
    else:
        pc = np.array(pc)

    lista = []
    for i in pc:
        lista.append([i[0], i[1]])

    idx = []
    lista_partenerts = []
    for id, i in enumerate(lista):
        a = i[0]
        b = i[1]
        list_2 = [i for i in zip(a, b)]
        sort_list = sorted(list_2)
        if sort_list not in lista_partenerts:
            lista_partenerts.append(sort_list)
            idx.append(id)

    print('parejas_iniciales', len(pc))
    pc = pc[idx]
    print('comparaciones', len(pc))
    return pc


def gen_rot_matrix_ref(residuos1, residuos2, parejas, vecs_center_cnclq_1, coord_conclq_2):
    """
    Genera matriz de rotacion con parejas
    :param residuos1: lista de objetos residuo de la proteina 1
    :param residuos2: lista de objetos residuo de la proteina 2
    :param parejas: parejas en formato de tupla en un lista [(1,1),...(89,90)]
    :param vecs_center_cnclq_1: vectores centricos previamente calculados
    :param coord_conclq_2: coordenadas originales de la proteina 2
    :return: coordenadas de proteina rotada y trasladada, proteina original, Matriz de rotacion, Baricentro de parejas.
    """
    # aveces truena si los pdbs no tienen los numeros de residuos continuos.
    coord_new_1 = [[res.GetAtom('CA').coord for res in residuos1 if i[0] == res.resx] for i in parejas]
    coord_new_2 = [[res.GetAtom('CA').coord for res in residuos2 if i[1] == res.resx] for i in parejas]

    coord_new_1 = np.array([y for x in coord_new_1 for y in x], dtype=np.float)
    coord_new_2 = np.array([y for x in coord_new_2 for y in x], dtype=np.float)

    bari_new_1 = coord_new_1.mean(0)
    bari_new_2 = coord_new_2.mean(0)

    vecs_center_1 = coord_new_1 - bari_new_1
    vecs_center_2 = coord_new_2 - bari_new_2

    # Se genera matriz de rotacion con vectores centricos de parejas anteriores
    matriz_R = matrix_R(vecs_center_1, vecs_center_2)
    matriz_rotacion = rotation_matrix(matriz_R)
    # se aplica matriz de rotacion a coordenadas de proteina a rotar
    # la proteina consiste en solo carbonos alfa
    vector_rotado = rotation_vectors(vecs_center_cnclq_1, matriz_rotacion)

    protein_trasladado_rotado = vector_rotado + bari_new_2

    protein_to_compare = coord_conclq_2

    return (protein_trasladado_rotado, protein_to_compare,matriz_rotacion, bari_new_2)
