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
import math
#lectura de paso final
import pandas as pd
# por si no jala
import os
os.chdir('/home/serch/pdbmani/Serch')

# lectura de archivo
file1 = '/home/serch/pdbmani/Serch/pdbs/1zao.pdb'  # sys.argv[1] #2fmq.pdb
file2 = '/home/serch/pdbmani/Serch/pdbs/1kj9.pdb'  # sys.argv[2] #2bpt.pdb

# file1 = 'pdbs/2mhu.pdb'  # sys.argv[1]
# file2 = 'pdbs/2mrt.pdb'  # sys.argv[2]

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

# creando tabla de estructura secundaria para filtro de SS
ss1 = fc.create_ss_table(pdb11)
ss2 = fc.create_ss_table(pdb22)


def eval_dihedral(ang_ref, ang_tar, cutoff=30):
    """
    Evaluacion de los angulo dihedrales, manteniendo aquellos que presentan un cierto cutoff.
    :param ang_ref: angulo phi o psi
    :param ang_tar: angulo phi o psi
    :param cutoff: limite de diferencia de angulo para filtro
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
    # print(vecs_c_1[:3], vecs_c_2[:3])
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
    coord_rotado = [np.matmul(matriz_rotacion, coord_atom) for coord_atom in vector_gorro]

    return (np.array(coord_rotado))


pc = pd.read_pickle('parejas_alineables_1zao_1kj9.pkl').values
pc = np.array([i[0] for i in pc])

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

cliques_finales = []
for cand_1,cand_2, mat_rot in pc:
    cliques_finales.append([cand_1,cand_2])

print(cliques_finales)

res_conclq_1 = [res for res in pdb11]
res_conclq_2 = [res for res in pdb22]

atom_conclq_1 = [res.GetAtom('CA') for res in pdb11]
atom_conclq_2 = [res.GetAtom('CA') for res in pdb22]

coord_conclq_1 = np.array([res.coord for res in atom_conclq_1], dtype=np.float)
coord_conclq_2 = np.array([res.coord for res in atom_conclq_2], dtype=np.float)

bari_con_clq_1 = coord_conclq_1.mean(0)
bari_con_clq_2 = coord_conclq_2.mean(0)

vecs_center_cnclq_1 = coord_conclq_1 - bari_con_clq_1
vecs_center_cnclq_2 = coord_conclq_2 - bari_con_clq_2

number_of_residues_final = len(res_conclq_1)

val = 0
so_winner = 0.0
candidatos = []
winner_matrix_rotation = []
winner_baricenter = []
print('numero de comparaciones', len(pc))


def gen_rot_matrix_ref(parejas):
    """
    Genera la matriz de rotacion por medio de las coordendas de las parejas seleccionadas
    :param parejas:
    :return: proteina rotada y trasladada, proteina a comparar, matriz de rotacion, baricentro de parejas
    """
    coord_new_1 = [[res.GetAtom('CA').coord for res in res_conclq_1 if i[0] == res.resi] for i in parejas]
    coord_new_2 = [[res.GetAtom('CA').coord for res in res_conclq_2 if i[1] == res.resi] for i in parejas]

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


def fun_resiudos_match(protein_trasladado_rotado, protein_to_compare, res_1, res_2):
    """
    genera los Match de parejas de residuos que cumplen un treshold de distancia (RMSD)
    :param protein_trasladado_rotado:
    :param protein_to_compare:
    :param res_1:
    :param res_2:
    :return: Lista ordenada de parejas de residuos y su distancia
    """
    return sorted([[math.sqrt(sum((c_2 - c_1) ** 2)), (res1.resi, res2.resi)] for c_1, res1 in zip(
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


# cosas que tengo que declarar para que no me diga nada pycharm...
matriz_rotacion = []
bari_new_2 = []
winner_parejas = []

for cand_1, cand_2, mat_rot in pc:

    # if set(cand_1).issubset([107, 106, 108, 109, 110, 112, 111]) and set(
    #         cand_2).issubset([139, 138, 140, 141, 142, 144, 143]):
        #primera iteracion sin cliques se aplica la matriz de rotacion y baricentro
        print('***********************************************************')
        print(val, cand_1, cand_2)
        res_sinclq_1 = [res for res in pdb11 if res.resi not in cand_1]
        res_sinclq_2 = [res for res in pdb22 if res.resi not in cand_2]

        coord_sinclq_1 = np.array([res.GetAtom('CA').coord for res in res_sinclq_1], dtype=np.float)
        coord_sinclq_2 = np.array([res.GetAtom('CA').coord for res in res_sinclq_2], dtype=np.float)

        bari_1 = coord_sinclq_1.mean(0)
        bari_2 = coord_sinclq_2.mean(0)

        vecs_center_1 = coord_sinclq_1 - bari_1
        # aplico matriz de rotacion de cliques a vectores centricos sin clique
        vector_rotado = rotation_vectors(vecs_center_1, mat_rot)
        protein_trasladado_rotado = vector_rotado + bari_2

        protein_to_compare = coord_sinclq_2

        # apilo la distancia y la pareja de residuos correspondientes si cumple con que el RMSD sea menor a 3.5
        residuos_match = fun_resiudos_match(protein_trasladado_rotado, protein_to_compare,
                                            res_sinclq_1, res_sinclq_2)
        # filtro parejas
        cand_n = filter_pairs(residuos_match, flag=False)
        # calculo el SO
        so_temp = round(len(cand_n) / (number_of_residues_final - 7), 4)
        print('PRE_SO:', so_temp)
        so_temp_plus_1 = 0.0

        # Refinamiento por medio de las parejas seleccionadas y el clique.
        while so_temp_plus_1 < so_temp:  # Primer refinamiento
            parejas = [i[1] for i in cand_n]
            for i, j in zip(cand_1, cand_2):
                parejas.insert(0, (i, j))

            # aqui comienza el segundo alineamiento!! Refinamiento
            ptr, ptc, mr, bc = gen_rot_matrix_ref(parejas)
            # match residuos ordenado por distancia
            rm = fun_resiudos_match(ptr, ptc, res_conclq_1, res_conclq_2)

            # quitar residuos repetidos
            cand_n = filter_pairs(rm, flag=False)
            so_temp_plus_1 = round(len(cand_n) / number_of_residues_final, 4)
            so_temp_minus_1 = so_temp
            print(so_temp_plus_1)
            if so_temp_plus_1 < so_temp:  # evita infinite loop
                break

            print(so_temp_minus_1, so_temp_plus_1)

            # Rerefinamiento por si puede ir encontrando nuevas y mejores parejas
            while so_temp_minus_1 < so_temp_plus_1:  # segundo refinamiento iterativo
                so_temp_minus_1 = so_temp_plus_1
                parejas = [i[1] for i in cand_n]
                for i, j in zip(cand_1, cand_2):
                    parejas.insert(0, (i, j))

                # aqui comienza el segundo alineamiento!! Refinamiento
                ptr, ptc, matriz_rotacion, bari_new_2 = gen_rot_matrix_ref(parejas)
                # match residuos ordenado por distancia
                rm = fun_resiudos_match(ptr, ptc, res_conclq_1, res_conclq_2)

                # quitar residuos repetidos
                cand_n = filter_pairs(rm, flag=False)
                so_temp_plus_1 = round(len(cand_n) / number_of_residues_final, 4)

                print(so_temp_minus_1, so_temp_plus_1)

            # actualizacion de datos
            if so_temp_plus_1 < so_temp_minus_1:
                so_temp_plus_1 = so_temp_minus_1
            # actualizacion de datos

            if so_temp_plus_1 > so_temp:
                so_temp = so_temp_plus_1

        # check que si este guardando el SO
        print(so_winner)

        # Si supera el SO ganador se guardan los parametros y se actualiza el SO
        if so_temp > so_winner:

            so_winner = so_temp  # actualizacion so
            winner_matrix_rotation = matriz_rotacion  # actualizacion mr
            winner_baricenter = bari_new_2  # actualizacion bc
            candidatos = [cand_1, cand_2]  # actualizacion de parejas de cliques estrella
            winner_parejas = cand_n   # actualizacion de parejas y distancia.
            print('========================='*3)
            print(so_temp_plus_1)
            print('cliques', candidatos)
            print('numero de parejas', len(cand_n))
            print('iteracion %s' % val, 'SO: %1.4f' % so_temp)
            print('RMSD:', np.mean([x[0] for x in cand_n]))
            print('parejas:', [x[1] for x in cand_n])
            print('================================================================================')

        val = val+1

        if so_temp == 1:
            break

print('=====pareja ganadora======')
print('cliques', candidatos)
print('numero de parejas', len(winner_parejas))
print('SO: %1.4f' % so_winner)
print('RMSD:', np.mean([x[0] for x in winner_parejas]))
print('parejas:', sorted([x[1] for x in winner_parejas]))

# #prueba para saber si los CA si estan rotando y trasladando bien
# coord_protein_1 = np.array([res.GetAtom('CA').coord for res in pdb11], dtype=np.float)
# bari_full_1 = coord_protein_1.mean(0)
# vecs_center_protein_1 = coord_protein_1 - bari_full_1
#
# # aplicacion de la rotacion y traslacion a toda la proteina
# vector_rotado = fc.rotation_vectors(vecs_center_protein_1, winner_matrix_rotation)
# protein_trasladado_rotado = vector_rotado + winner_baricenter

# Actualizacion de coordendas
coord_protein_1 = np.array([res.GetAtom(name).coord for res in pdb11 for name in res.atomnames],
                           dtype=np.float)
bari_full_1 = coord_protein_1.mean(0)
vecs_center_protein_1 = coord_protein_1 - bari_full_1

# aplicacion de la rotacion y traslacion a toda la proteina
vector_rotado = rotation_vectors(vecs_center_protein_1, winner_matrix_rotation)
protein_trasladado_rotado = vector_rotado + winner_baricenter

# actualizacion de coordenadas
k = 0
for res in pdb11:
    for atom in res.atomnames:
        setattr(res.GetAtom(atom), 'coord', protein_trasladado_rotado[k])
        k = k+1


# se escribe el nuevo pdb rotado y trasladado
pdb1.WriteToFile(file_out_name=file1[-8:-4]+'_vs_'+file2[-8:-4]+'_'+str(datetime.datetime.now())[:19])

#tiempo de ejecucion
timenow = datetime.datetime.now()
print('Tiempo Total:', timenow - time_all)
