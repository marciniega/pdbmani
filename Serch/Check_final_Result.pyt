#!/usr/bin/env python
# coding: utf-8
str1_1xxa_clean = [
82,
85,
86,
87,
88,
89,
90,
91,
92,
93,
95,
96,
97,
98,
99,
100,
101,
102,
103,
104,
106,
107,
108,
109,
110,
111,
112,
113,
114,
115,
116,
117,
119,
121,
122,
123,
124,
125,
126,
127,
128,
129,
130,
132,
134,
135,
136,
137,
138,
140,
141,
142,
143,
144,
145,
146,
147,
148,
149,
150,
151,
152]


# In[2]:


str2_1tig_clean = [
   95,
    91,
    90,
    89,
    88,
    87,
    86,
    85,
    84,
    83,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    133,
    92,
    134,
    136,
    137,
    140,
    138,
    139,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    165,
    153,
    163,
    155,
    161,
    160,
    162,
    164,
    166,
    167,
    168,
    169,
    170,
    113,
    114,
    112,
    111,
    110,
    109,
    108,
    107,
    106,
    105,
    104,
    103,
    102
]


# In[5]:


# parejas respuesta final
lista = []
for i in zip(str1_1xxa_clean,str2_1tig_clean):
    lista.append(i)


# In[6]:


lista


# In[7]:


# mis parejas
lista2 = [
    (83, 94),
    (84, 93),
    (85, 92),
    (86, 91),
    (88, 89),
    (89, 88),
    (90, 87),
    (91, 86),
    (92, 85),
    (93, 84),
    (95, 116),
    (96, 117),
    (97, 118),
    (98, 119),
    (99, 120),
    (102, 129),
    (103, 130),
    (104, 132),
    (105, 133),
    (106, 134),
    (107, 135),
    (108, 136),
    (109, 137),
    (110, 138),
    (111, 139),
    (112, 140),
    (113, 142),
    (114, 143),
    (115, 144),
    (116, 145),
    (117, 146),
    (120, 147),
    (121, 148),
    (122, 149),
    (123, 150),
    (125, 153),
    (126, 154),
    (127, 155),
    (129, 161),
    (130, 162),
    (131, 163),
    (132, 164),
    (133, 165),
    (134, 166),
    (135, 167),
    (136, 168),
    (137, 169),
    (138, 170),
    (139, 111),
    (140, 113),
    (141, 114),
    (142, 109),
    (143, 108),
    (144, 106),
    (146, 105),
    (147, 104),
    (148, 103),
    (149, 101),
    (151, 100),
    (152, 99),
]


# In[9]:


# numero de parejas que yo obtengo
len([c for c in lista2 if c in lista])


# In[11]:


#numero de parejas de la respuesta
len(lista)


# In[20]:


# porcentaje de parejas mias que estan en la respuesta
perc = (len([c for c in lista2 if c in lista])/len(lista)) *100
print('porcentaje % 1.3f'  % perc)


# # Checar que 7-cliques se forman y encontrar la pareja de cliques que forman el alineamiento
# Con el fin de buscar en que parte de nuestro algoritmo no se esta formando esa pareja de cliques

# In[25]:


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

pdb11 = [res for res in pdb11 if res.resi in str1_1xxa_clean]
pdb22 = [res for res in pdb22 if res.resi in str2_1tig_clean]

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


# In[27]:


enlaces1 = (get_df_distancias(pdb11))
enlaces2 = (get_df_distancias(pdb22))

red1 = (nx.Graph(enlaces1))
red2 = (nx.Graph(enlaces2))

cliques_1, cliques_max_1 = fc.gen_cliques(red1, k=7)
cliques_2, cliques_max_2 = fc.gen_cliques(red2, k=7)


# In[48]:


import pandas as pd
df = pd.DataFrame(cliques_max_1)


# In[49]:


mask = np.where(df[7].isnull(),True,False)
df = df[mask].reset_index(drop=True).drop([7,8],1)


# In[50]:


df


# In[51]:


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


# In[52]:


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


# In[53]:


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


# In[54]:


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


# In[55]:


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


# In[56]:


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


# In[57]:


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


# In[59]:


parejas_cliques_finales[0]


# In[60]:


pc = parejas_cliques_finales


# In[61]:


pc = np.array([i[0] for i in pc])


# In[62]:


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


# In[69]:


pc[:4]


# In[65]:


cliques_finales = []
for cand_1,cand_2, mat_rot in pc:
    cliques_finales.append([cand_1,cand_2])


# In[66]:


cliques_finales


# In[67]:


mis_cliques_finales = [
    [[141, 146, 144, 145, 147, 142, 143], [130, 135, 133, 134, 136, 131, 132]],
    [[141, 146, 144, 145, 147, 142, 143], [135, 130, 132, 131, 129, 134, 133]],
    [[141, 142, 144, 145, 143, 147, 146], [132, 133, 135, 136, 134, 138, 137]],
    [[141, 142, 145, 144, 143, 147, 146], [129, 130, 133, 132, 131, 135, 134]],
    [[141, 142, 143, 144, 145, 147, 146], [133, 134, 135, 136, 137, 139, 138]],
    [[146, 144, 145, 147, 148, 149, 150], [134, 132, 133, 135, 136, 137, 138]],
    [[146, 144, 145, 147, 143, 148, 149], [132, 134, 133, 131, 135, 130, 129]],
    [[146, 144, 145, 147, 141, 142, 143], [133, 135, 134, 132, 138, 137, 136]],
    [[146, 144, 145, 147, 143, 148, 149], [133, 135, 134, 132, 136, 131, 130]],
    [[146, 144, 145, 147, 148, 150, 149], [133, 135, 134, 132, 131, 129, 130]],
    [[146, 144, 145, 147, 143, 148, 149], [135, 133, 134, 136, 132, 137, 138]],
    [[146, 144, 145, 147, 148, 150, 149], [135, 133, 134, 136, 137, 139, 138]],
    [[146, 144, 147, 145, 143, 148, 149], [132, 130, 133, 131, 129, 134, 135]],
    [[146, 144, 147, 145, 148, 150, 149], [132, 130, 133, 131, 134, 136, 135]],
    [[146, 145, 147, 144, 143, 148, 149], [133, 132, 134, 131, 130, 135, 136]],
    [[146, 145, 147, 148, 149, 150, 151], [133, 132, 134, 135, 136, 137, 138]],
    [[146, 145, 147, 148, 149, 151, 150], [134, 133, 135, 136, 137, 139, 138]],
    [[146, 145, 147, 144, 141, 142, 143], [134, 135, 133, 136, 139, 138, 137]],
    [[146, 145, 147, 144, 148, 149, 150], [134, 135, 133, 136, 132, 131, 130]],
    [[146, 145, 147, 148, 149, 151, 150], [134, 135, 133, 132, 131, 129, 130]],
    [[142, 144, 145, 141, 143, 147, 135], [135, 133, 132, 136, 134, 130, 92]],
    [[142, 144, 145, 141, 143, 146, 147], [135, 133, 132, 136, 134, 131, 130]],
    [[142, 144, 145, 141, 143, 140, 135], [132, 134, 135, 131, 133, 129, 130]],
    [[144, 145, 143, 141, 142, 135, 140], [134, 133, 135, 137, 136, 154, 138]],
    [[144, 145, 143, 141, 142, 140, 135], [134, 133, 135, 137, 136, 139, 138]],
    [[144, 145, 143, 147, 146, 148, 149], [134, 135, 133, 137, 136, 138, 139]],
    [[110, 109, 106, 107, 112, 108, 111], [134, 133, 130, 131, 136, 132, 135]],
    [[110, 109, 106, 107, 108, 105, 111], [134, 133, 130, 131, 132, 129, 135]],
    [[110, 109, 106, 107, 112, 108, 111], [136, 135, 132, 133, 138, 134, 137]],
    [[110, 109, 106, 107, 108, 104, 105], [136, 135, 132, 133, 134, 130, 131]],
    [[110, 109, 112, 111, 106, 107, 108], [135, 136, 133, 134, 139, 138, 137]],
    [[110, 109, 112, 111, 106, 107, 108], [133, 132, 135, 134, 129, 130, 131]],
    [[110, 109, 107, 106, 108, 111, 105], [133, 134, 136, 137, 135, 132, 138]],
    [[110, 109, 107, 106, 108, 104, 105], [133, 134, 136, 137, 135, 139, 138]],
    [[110, 109, 107, 106, 108, 126, 104], [133, 134, 136, 137, 135, 154, 138]],
    [[110, 109, 107, 106, 105, 126, 108], [133, 134, 136, 137, 138, 154, 135]],
    [[110, 109, 107, 106, 112, 108, 111], [132, 133, 135, 136, 130, 134, 131]],
    [[110, 109, 107, 106, 108, 104, 105], [132, 133, 135, 136, 134, 138, 137]],
    [[110, 109, 107, 106, 108, 111, 105], [135, 134, 132, 131, 133, 136, 130]],
    [[110, 109, 107, 106, 108, 104, 105], [135, 134, 132, 131, 133, 129, 130]],
    [[110, 109, 108, 106, 112, 107, 111], [134, 135, 136, 138, 132, 137, 133]],
    [[110, 109, 108, 106, 107, 104, 105], [134, 135, 136, 138, 137, 140, 139]],
    [[110, 109, 108, 106, 111, 105, 107], [134, 135, 136, 138, 133, 139, 137]],
    [[110, 106, 107, 108, 109, 111, 105], [130, 134, 133, 132, 131, 129, 135]],
    [[110, 106, 107, 108, 109, 104, 105], [130, 134, 133, 132, 131, 136, 135]],
    [[110, 112, 111, 106, 107, 108, 109], [136, 134, 135, 140, 139, 138, 137]],
    [[109, 106, 107, 105, 108, 110, 111], [132, 135, 134, 136, 133, 131, 130]],
    [[109, 106, 107, 108, 110, 112, 111], [132, 135, 134, 133, 131, 129, 130]],
    [[109, 107, 108, 104, 105, 106, 100], [133, 135, 134, 138, 137, 136, 154]],
    [[109, 107, 108, 105, 106, 126, 104], [136, 134, 135, 132, 133, 92, 130]],
    [[109, 107, 108, 105, 106, 110, 111], [136, 134, 135, 132, 133, 137, 138]],
    [[109, 107, 108, 106, 112, 110, 111], [136, 134, 135, 133, 139, 137, 138]],
    [[106, 107, 108, 100, 104, 105, 103], [134, 135, 136, 92, 132, 133, 130]],
    [[106, 107, 108, 104, 105, 109, 110], [134, 135, 136, 132, 133, 137, 138]],
    [[106, 107, 108, 105, 109, 111, 110], [134, 135, 136, 133, 137, 139, 138]],
    [[106, 107, 108, 109, 112, 110, 111], [134, 135, 136, 137, 140, 138, 139]],
    [[106, 107, 108, 109, 110, 126, 104], [134, 135, 136, 137, 138, 154, 133]],
    [[110, 109, 106, 107, 112, 108, 111], [137, 138, 141, 140, 135, 139, 136]],
    [[110, 109, 106, 107, 108, 104, 105], [137, 138, 141, 140, 139, 143, 142]],
    [[110, 109, 106, 107, 112, 108, 111], [141, 140, 137, 138, 143, 139, 142]],
    [[110, 109, 106, 107, 108, 104, 105], [141, 140, 137, 138, 139, 135, 136]],
    [[110, 109, 106, 107, 112, 108, 111], [139, 140, 143, 142, 137, 141, 138]],
    [[110, 109, 106, 107, 108, 104, 105], [139, 140, 143, 142, 141, 145, 144]],
    [[110, 109, 106, 107, 111, 105, 108], [139, 140, 143, 142, 138, 144, 141]],
    [[110, 109, 106, 107, 112, 108, 111], [142, 141, 138, 139, 144, 140, 143]],
    [[110, 109, 106, 107, 108, 105, 111], [142, 141, 138, 139, 140, 137, 143]],
    [[110, 109, 105, 104, 106, 107, 108], [138, 139, 143, 144, 142, 141, 140]],
    [[110, 109, 105, 106, 107, 111, 108], [138, 139, 143, 142, 141, 137, 140]],
    [[110, 109, 107, 106, 108, 111, 105], [140, 139, 137, 136, 138, 141, 135]],
    [[110, 109, 107, 106, 108, 104, 105], [140, 139, 137, 136, 138, 134, 135]],
    [[110, 109, 107, 106, 112, 108, 111], [140, 141, 143, 144, 138, 142, 139]],
    [[110, 109, 107, 106, 108, 105, 111], [140, 141, 143, 144, 142, 145, 139]],
    [[110, 109, 107, 106, 112, 108, 111], [143, 142, 140, 139, 145, 141, 144]],
    [[110, 109, 107, 106, 108, 104, 105], [143, 142, 140, 139, 141, 137, 138]],
    [[110, 109, 107, 106, 111, 105, 108], [143, 142, 140, 139, 144, 138, 141]],
    [[110, 109, 108, 106, 112, 107, 111], [139, 138, 137, 135, 141, 136, 140]],
    [[110, 109, 108, 106, 107, 104, 105], [139, 138, 137, 135, 136, 133, 134]],
    [[110, 109, 108, 106, 111, 105, 107], [139, 138, 137, 135, 140, 134, 136]],
    [[110, 109, 108, 106, 112, 107, 111], [141, 142, 143, 145, 139, 144, 140]],
    [[109, 106, 105, 108, 107, 110, 111], [137, 140, 141, 138, 139, 136, 135]],
    [[109, 106, 105, 104, 108, 107, 110], [143, 140, 139, 138, 142, 141, 144]],
    [[109, 106, 105, 108, 107, 111, 110], [143, 140, 139, 142, 141, 145, 144]],
    [[106, 105, 107, 104, 108, 109, 110], [139, 140, 138, 141, 137, 136, 135]],
    [[106, 105, 107, 108, 109, 111, 110], [139, 140, 138, 137, 136, 134, 135]],
    [[106, 105, 107, 104, 108, 109, 110], [141, 140, 142, 139, 143, 144, 145]],
    [[106, 107, 108, 104, 105, 100, 103], [143, 142, 141, 145, 144, 148, 146]],
]


# In[73]:


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


# In[77]:

import math
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
    # residuos_match = reversed(residuos_match)

    print(residuos_match)

    print(sorted(residuos_match))

    exit()
    c1 = []
    c2 = []
    cand_n = []
    for i in residuos_match:
        if (i[1][0] in c1) or (i[1][1] in c2):
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
# print(len(protein_trasladado_rotado))
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
# pdb1.WriteToFile(file_out_name='1xxa_vs_1tig_click'+str(datetime.datetime.now())[:19])

#tiempo de ejecucion
timenow = datetime.datetime.now()
print('Tiempo Total:', timenow - time_all)

import math
