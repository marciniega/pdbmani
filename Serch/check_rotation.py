import numpy as np

vecs_1 = np.round(np.array([[57.57241322, 22.89437262,53.77215885],
 [54.47094521, 22.77946566, 55.89182311],
 [52.65864156, 26.03816172, 54.84401804]]),3)

vecs_2 = np.array([[55.956, 23.95,  54.11 ],
 [55.426, 21.004, 56.484],
 [53.32,  26.758, 53.914]])


# vecs_1 = np.array([[1,3,4],[5,1,6],[1,2,1]])
#
# vecs_2 = vecs_1 * -1

# print(vecs_1)

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
    return (np.array(matriz_R))

def rotation_matrix(matriz_R):
    """utilizando la funcion giant_matrix, fijando los valores de i,j
    se calcula la matriz de rotacion con los eigenvectores y eigenvalores
    arroja una matriz de rotacion que depende de la matriz gigante
    """
    eignvalues, eigenvectors = np.linalg.eig(matriz_R)
    q = eigenvectors[:, np.argmax(eignvalues)]

    # matriz de rotacion con eigenvectores forma USING QUATERNIONS TO CALCULATE RMSD
    # q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    # matriz_rotacion = np.array([
    #     [(q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
    #     [2 * (q1 * q2 + q0 * q3), (q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
    #     [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)]
    # ], dtype=np.float64)

    # matriz de rotacion con eigenvectores forma ON THE RTHOGONAL TRANSFORMATION FOR STRUCTURAL COMPARISON
    q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
    matriz_rotacion = np.array([
        [(q1 ** 2 + q2 ** 2 - q3 ** 2 - q4 ** 2), 2 * (q2 * q3 + q1 * q4), 2 * (q2 * q4 - q1 * q3)],
        [2 * (q2 * q3 - q1 * q4), (q1 ** 2 + q3 ** 2 - q2 ** 2 - q4 ** 2), 2 * (q3 * q4 + q1 * q2)],
        [2 * (q2 * q4 + q1 * q3), 2 * (q3 * q4 - q1 * q2), (q1 ** 2 + q4 ** 2 - q2 ** 2 - q3 ** 2)]
    ], dtype=np.float64)

    return (matriz_rotacion)


def rotation_vectors(vector_gorro, matriz_rotacion):
    """obtencion de vector rotado,
    utilizando la matriz de rotacion
    y los vectores gorro a rotar y trasladar"""
    coord_rotado = [np.matmul(matriz_rotacion, coord_atom) for coord_atom in vector_gorro]

    return (coord_rotado)


def rmsd_between_cliques(clique_trasladado_rotado, atom_to_compare):
    """Calculo de rmsd entre cliques tomando el atomo rotado y trasladado
    y el atomo a comparar, por el momento solo imprime el resultado"""
    # primer RMSD entre atomos
    pre_rmsd = np.sum((np.array(atom_to_compare, dtype=np.float64) - clique_trasladado_rotado) ** 2, 1)
    rmsd_i = lambda i: np.sqrt(i) / 3
    rmsd_final = rmsd_i(pre_rmsd).mean()

    return (rmsd_final)


def align(c_1, c_2):

    bari_1 = c_1.mean(0)
    bari_2 = c_2.mean(0)
    print("baricentro1",bari_1)
    print("baricentro2", bari_2)
    vecs_center_1 = c_1 - bari_1
    vecs_center_2 = c_2 - bari_2
    print("VECS_CENTER1", vecs_center_1)
    print("VECS_CENTER2", vecs_center_2)
    matriz_R = matrix_R(vecs_center_1, vecs_center_2)
    matriz_rotacion = rotation_matrix(matriz_R)
    print(matriz_rotacion)
    vector_rotado = rotation_vectors(vecs_center_1, matriz_rotacion)
    print(vector_rotado)
    # vector_rotado_trasladado_a_clique2 = vector_rotado + bari_2
    # print(vector_rotado_trasladado_a_clique2)
    # protein_trasladado_rotado = vector_rotado_trasladado_a_clique2
    # protein_to_compare = np.array(c_2, dtype=np.float)

    pre_rmsd = np.sum((np.array(
        vecs_center_2, dtype=np.float64) - vector_rotado) ** 2, 1)
    rmsd_i = lambda i: np.sqrt(i) / 3

    rmsd_final = rmsd_i(pre_rmsd).mean(0)
    # TE PUEDES AHORRAR EL PASO DE TRASLADAR SI CALCULAS EN LOS VECTORES CENTRICOS.
    rmsd_final = rmsd_between_cliques(vector_rotado, vecs_center_2)

    return rmsd_final

print("vec1:",vecs_1,"\n","vecs_2",vecs_2)

print("RMSD", align(vecs_1,vecs_2))