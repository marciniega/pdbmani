#!/usr/bin/env python
import numpy as np
import read_pdb_tools as rpt
import math_vect_tools as mvt

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

def check_cliques_rmsd( c_1 , c_2 ):
    pre_rotation = matrix_R(c_1, c_2)
    rot_mat = rotation_matrix(pre_rotation)
    return rmsd(rotation_vectors(c_1, rot_mat) , c_2)

def rmsd(mov, tar):
    """Calculo de rmsd entre cliques tomando el atomo rotado y trasladado
    y el atomo a comparar, por el momento solo imprime el resultado"""
    pre_rmsd = (np.sum((mov - tar) ** 2, 1)).mean()
    rmsd_final = np.sqrt(pre_rmsd)
    return (rmsd_final)

def rotation_angles(matriz_R,degrees=False):
    eignvalues, eigenvectors = np.linalg.eig(matriz_R)
    q = eigenvectors[:, np.argmax(eignvalues)]
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    phi = np.arctan2(2*((q0*q1)+(q2*q3)),1-2*(q1**2 +q2**2))
    theta = np.arcsin(2*(q0*q2-q3*q1))
    psi = np.arctan2(2*((q0*q3)+(q1*q2)),1-2*(q2**2+q3**2))
    if degrees:
       phi = phi*(180/np.pi)
       theta = theta*(180/np.pi)
       psi = psi*(180/np.pi)
    return np.array([phi,theta,psi])

