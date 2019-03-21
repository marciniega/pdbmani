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
file1 = '/home/serch/pdbmani/Serch/pdbs/1xxa_clean.pdb'  # sys.argv[1]
file2 = '/home/serch/pdbmani/Serch/pdbs/1tig_clean.pdb'  # sys.argv[2]

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


pc = pd.read_pickle('parejas_alineables.pkl').values
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
so = 0
candidatos = []
winner_matrix_rotation = []
winner_baricenter = []
print('numero de comparaciones', len(pc))

for cand_1, cand_2, mat_rot in pc:

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
    residuos_match = [[round(math.sqrt(sum((c_2 - c_1) ** 2)), 5), (res1.resi, res2.resi)] for c_1, res1 in zip(
        protein_trasladado_rotado, res_sinclq_1) for c_2, res2 in zip(
        protein_to_compare, res_sinclq_2) if math.sqrt(sum((c_2 - c_1) ** 2)) < 3.5]

    residuos_match = sorted(residuos_match)

    c1 = []
    c2 = []
    cand_n = []
    for i in residuos_match:
        if (i[1][0] in c1) or (i[1][1] in c2) or (i[0] > 3.5):
            continue
        else:
            c1.append(i[1][0])
            c2.append(i[1][1])
            cand_n.append(i)

    so_temp = len(cand_n) / (number_of_residues_final - 7)

    # print(len(protein_trasladado_rotado))
    # print(protein_trasladado_rotado[:10])
    # print('================================================================================')

    # se agrega emparejamiento de cliques a las parejas anteriormente generadas
    parejas = [i[1] for i in cand_n]
    for i, j in zip(cand_1, cand_2):
        parejas.insert(0, (i, j))

    parejas = [(82, 95),
     (85, 91),
     (86, 90),
     (87, 89),
     (88, 88),
     (89, 87),
     (90, 86),
     (91, 85),
     (92, 84),
     (93, 83),
     (95, 115),
     (96, 116),
     (97, 117),
     (98, 118),
     (99, 119),
     (100, 120),
     (101, 121),
     (102, 122),
     (103, 133),
     (104, 92),
     (106, 134),
     (107, 136),
     (108, 137),
     (109, 140),
     (110, 138),
     (111, 139),
     (112, 141),
     (113, 142),
     (114, 143),
     (115, 144),
     (116, 145),
     (117, 146),
     (119, 147),
     (121, 148),
     (122, 149),
     (123, 150),
     (124, 165),
     (125, 153),
     (126, 163),
     (127, 155),
     (128, 161),
     (129, 160),
     (130, 162),
     (132, 164),
     (134, 166),
     (135, 167),
     (136, 168),
     (137, 169),
     (138, 170),
     (140, 113),
     (141, 114),
     (142, 112),
     (143, 111),
     (144, 110),
     (145, 109),
     (146, 108),
     (147, 107),
     (148, 106),
     (149, 105),
     (150, 104),
     (151, 103),
     (152, 102)]

    # aqui comienza el segundo alineamiento!!

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

    residuos_match = [[math.sqrt(sum((c_2 - c_1) ** 2)), (res1.resi, res2.resi)] for c_1, res1 in zip(
        protein_trasladado_rotado, res_conclq_1) for c_2, res2 in zip(
        protein_to_compare, res_conclq_2) if math.sqrt(sum((c_2 - c_1) ** 2)) < 3.5]

    # residuos_match = sorted(residuos_match)

    c1 = []
    c2 = []
    cand_n = []
    for i in residuos_match:
        if (i[1][0] in c1) or (i[1][1] in c2) or (i[0] > 3.5):
            continue
        else:
            c1.append(i[1][0])
            c2.append(i[1][1])
            cand_n.append(i)

    so_temp = len(cand_n) / number_of_residues_final

    if so_temp > so:

        so = so_temp
        candidatos = [cand_1, cand_2]
        winner_matrix_rotation = matriz_rotacion
        winner_baricenter = bari_new_2

        print('========================='*3)
        print('cliques', candidatos)
        print('numero de parejas', len(cand_n))
        print('iteracion %s' % val, 'SO: %1.4f' % so_temp)
        print('RMSD:', np.mean([x[0] for x in cand_n]))
        print('parejas:', [x[1] for x in cand_n])

        print(bari_new_1, bari_new_2)
        print(len(protein_trasladado_rotado))
        print(protein_trasladado_rotado[:10])
        print('================================================================================')

    val = val+1

    if so_temp == 1:
        break


# print(so,candidatos,mat_rot_winner,winner_baricenter)
# #prueba para saber si los CA si estan rotando y trasladando bien
# coord_protein_1 = np.array([res.GetAtom('CA').coord for res in pdb11], dtype=np.float)
# bari_full_1 = coord_protein_1.mean(0)
# vecs_center_protein_1 = coord_protein_1 - bari_full_1
#
# # aplicacion de la rotacion y traslacion a toda la proteina
# vector_rotado = fc.rotation_vectors(vecs_center_protein_1, mat_rot_winner)
# protein_trasladado_rotado = vector_rotado + winner_baricenter
#
# print(len(protein_trasladado_rotado

# print(protein_trasladado_rotado[:10])

#  Actualizacion de coordendas
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
pdb1.WriteToFile(file_out_name='1xxa_vs_1tig_'+str(datetime.datetime.now())[:19])

#tiempo de ejecucion
timenow = datetime.datetime.now()
print('Tiempo Total:', timenow - time_all)