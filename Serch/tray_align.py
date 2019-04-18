#!/bin/sh

# librerias que utilizaras
import numpy as np
# por si no te lee las tools o functions creadas
import sys
sys.path.append("/home/serch/pdbmani/Serch/math_tricks/")
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
# formato resultados
import pandas as pd
np.random.seed(8)
# lectura de archivo
# file1 = sys.argv[1]
# file2 = sys.argv[2]

# borrar esto despues
file1 = '/home/serch/pdbmani/Serch/pdbs/dimer_con_cofact_clean2.pdb'
file2 = '/home/serch/pdbmani/Serch/pdbs/tetra_con_cofact_clean2_click.pdb'

# global_cutoff = float(sys.argv[3])

# carpeta destino
directory = '/home/serch/pdbmani/Serch/pdbs/'

# Ya moficamos la cadenaa darle atomos
chain1 = 'A'
chain2 = 'A'

global_cutoff = 7  # modificar si quieres cambiar de cutoff!!! en filtro dihedral

# numero de cliques, preguntar en el software para generalizarlo INPUT...
number_elements_clique = 3

# se define la estructura
trj1 = rpt.Trajectory(file1)
trj2 = rpt.Trajectory(file2)
# se lee el pdb y se agrega al objeto
trj1.ReadTraj("%s" % file1)
trj2.ReadTraj("%s" % file2)

trj11 = trj1.frames
trj22 = trj2.frames

num_samples1 = int(len(trj11) * 0.05)
num_samples2 = int(len(trj22) * 0.05)

num_samples = num_samples1
if num_samples1 > num_samples2:
    num_samples = num_samples2

print('numero de muestras', num_samples)

# revisar si importa el orden.
trj11 = np.random.choice(trj11, size=num_samples, replace=False)
trj22 = np.random.choice(trj22, size=num_samples, replace=False)

#filtro SS
# no se puede todo de jalon primero sacarlo....

for inde, pdb in enumerate(zip(trj11, trj22)):

    number_elements_clique = 3

    print(inde, pdb)
    # first_align
    pdb1 = pdb[0]
    pdb1.Set_SS(dssp_file='pdbs/dimer_con_cofact_clean2')
    pdb1.SetDiheMain(chain_n=chain1)

    pdb2 = pdb[1]
    pdb2.Set_SS(dssp_file='pdbs/tetra_con_cofact_clean2_click')
    pdb2.SetDiheMain(chain_n=chain2)

    # se obtienen los residuos que perteneces a la cadena de interes por default chain = 'A'
    pdb11 = pdb1.GetResChain(chain=chain1)
    pdb22 = pdb2.GetResChain(chain=chain2)

    if len(pdb11) > 500 or len(pdb22) > 500:
        print('Esta muy grande la proteina va a tardar siglos ya ni le intentes!')
        print(file1, file2)
        pdb_file = open(directory+file1[-10:-4]+'_'+file2[-10:-4]+".delete", "w")
        pdb_file.write("Proteinas muy grandes, muchos residuos")
        pdb_file.close()
        exit()

    #verifica tamanio
    pdb1, pdb2, pdb11, pdb22 = fc.verify_file_order(pdb1, pdb2, pdb11, pdb22)

    # creando tabla de estructura secundaria para filtro de SS
    ss1 = fc.create_ss_table(pdb11)
    ss2 = fc.create_ss_table(pdb22)
    # print(ss1,ss2)
    # generacion de enlaces
    enlaces1 = (fc.get_df_distancias(pdb11))
    enlaces2 = (fc.get_df_distancias(pdb22))

    red1 = (nx.Graph(enlaces1))
    red2 = (nx.Graph(enlaces2))

    cliques_1, cliques_max_1 = fc.gen_cliques(red1, k=7)
    cliques_2, cliques_max_2 = fc.gen_cliques(red2, k=7)

    lenght_cliquemax_1 = len(cliques_max_1)
    lenght_cliquemax_2 = len(cliques_max_2)

    print("numero de cliques maximales combinaciones", lenght_cliquemax_1 * lenght_cliquemax_2)

    ####################################
    # Alineamiento por posicion del pdb#
    ####################################
    list_candidates = []
    for clique1 in cliques_max_1:
        res_clq_1 = [pdb1.GetResIdx(j) for j in clique1]
        for clique2 in cliques_max_2:
            res_clq_2 = [pdb2.GetResIdx(j) for j in clique2]

            # Filtro PHI PSI
            val_vec = []
            for res1 in res_clq_1:
                phi_ref = res1.phi
                psi_ref = res1.psi
                val = 0
                for res2 in res_clq_2:
                    phi_tar = res2.phi
                    psi_tar = res2.psi
                    if fc.eval_dihedral(phi_ref, phi_tar, cutoff=global_cutoff) and (
                            fc.eval_dihedral(psi_ref, psi_tar, cutoff=global_cutoff)):
                        val = val + 1
                val_vec.append(val)
            if val_vec.count(0) < 1:
                # debugg de vector de angulos dihedrales
                list_candidates.append([clique1, clique2])

    # # numero de candidatos y parejas generadas.
    print("numero de candidatos despues de filtro dihedral", len(list_candidates))
    print(list_candidates[:3])

    # filtro de tardara mucho
    if len(list_candidates) > 3000:
        print('no se filtraron los suficientes cambia el cutoff!')
        print(file1, file2)
        pdb_file = open(directory+file1[-10:-4]+'_'+file2[-10:-4]+".delete", "w")
        pdb_file.write("bajar_cutoff_dihedral")
        pdb_file.close()
        exit()

    if len(list_candidates) == 0:
        print('no hubo candidatos cambia filtro dihedral')
        exit()

    # GENERACION DE CLIQUES DE LA LISTA DE CLIQUES MAXIMALES APLICANDO FILTRO DIHEDRAL
    cliques_1_temp = []
    for clique1 in list_candidates:
        list_candidate = fc.gen_cliques_3(clique1[0])
        if list_candidate not in cliques_1_temp:
            cliques_1_temp.append(list_candidate)

    cliques_2_temp = []
    for clique2 in list_candidates:
        list_candidate = fc.gen_cliques_3(clique2[1])
        if list_candidate not in cliques_2_temp:
            cliques_2_temp.append(list_candidate)


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
                    res_clq_1 = [pdb1.GetResIdx(clq) for clq in clique1]
                    for clique2 in pareja[1]:
                        res_clq_2 = [pdb2.GetResIdx(clq) for clq in clique2]
                        if fc.score_ss(res_clq_1, res_clq_2):
                            coord_1 = np.array([res.GetAtom('CA').coord for res in res_clq_1])
                            coord_2 = np.array([res.GetAtom('CA').coord for res in res_clq_2])
                            if fc.align(coord_1, coord_2) < restriccion_rmsd:
                                cliques_candidate.append([clique1, clique2])

        else:
            for clique1 in cliques_1_align:
                res_clq_1 = [pdb1.GetResIdx(clq) for clq in clique1]
                for clique2 in cliques_2_align:
                    res_clq_2 = [pdb2.GetResIdx(clq) for clq in clique2]
                    if fc.score_ss(res_clq_1, res_clq_2):
                        coord_1 = np.array([res.GetAtom('CA').coord for res in res_clq_1])
                        coord_2 = np.array([res.GetAtom('CA').coord for res in res_clq_2])
                        if number_elements_clique == 7:
                            rmsd, mat_rot = fc.align(coord_1, coord_2, number_elements_clique=number_elements_clique)
                            if rmsd < restriccion_rmsd:
                                cliques_candidate.append([clique1, clique2, mat_rot])

                        else:
                            if fc.align(coord_1, coord_2) < restriccion_rmsd:
                                cliques_candidate.append([clique1, clique2])

        return cliques_candidate


    print('================ alineamiento de 3-clique y agrego el 4 elemento ===================')

    new_df_cliques1 = cliques_1_temp
    new_df_cliques2 = cliques_2_temp
    print(new_df_cliques1[0])
    print(new_df_cliques2[0])
    # if inde > 0:
    #     new_df_cliques1 = [y for x in cliques_1_temp for y in x]
    #     new_df_cliques2 = [y for x in cliques_2_temp for y in x]

    # emparejamiento de cliques
    cliques_candidatos = iter_align(number_elements_clique, new_df_cliques1, new_df_cliques2)

    # filtro de candidatos repetidos
    cliques_candidatos = fc.filter_candidates(cliques_candidatos, flag=False)

    # se agrega un elemento a cada pareja de cliques
    new_df_cliques = fc.add_element_to_clique(cliques_candidatos, cliques_max_1, cliques_max_2)

    number_elements_clique = number_elements_clique + 1

    print("candidatos con n-cliques, n =", number_elements_clique-1,
          "numero de parejas", len(cliques_candidatos))

    print(number_elements_clique, len(new_df_cliques))

    print('==================ya acabo con 3-cliques va con 4=========================')
    # refiltro
    # new_df_cliques = fc.filter_candidates(new_df_cliques, flag=False)

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
    print(cliques_temp[:1])
    print(len(cliques_temp))

    print(cliques_temp_add[:1])
    print(len(cliques_temp_add))

    print('==================ya acabo con 4-cliques va con 5=========================')

    cliques_temp_add_0 = [y for x in cliques_temp_add for y in x]

    # cliques_temp_add_0 = fc.filter_candidates(cliques_temp_add_0, flag=False)

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

    print(cliques_temp1[:1])
    print(len(cliques_temp1))

    print(cliques_temp1_add[:1])
    print(len(cliques_temp1_add))
    print('==================ya acabo con 5-cliques va con 6=========================')

    cliques_temp_add_11 = [y for x in cliques_temp1_add for y in x]

    # cliques_temp_add_11 = fc.filter_candidates(cliques_temp_add_11, flag=True)

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

    print(cliques_temp2[:1])
    print(len(cliques_temp2))

    print(cliques_temp2_add[:1])
    print(len(cliques_temp2_add))
    print('==================ya acabo con 6-cliques va con 7=========================')

    cliques_temp_add_22 = [y for x in cliques_temp2_add for y in x]
    # cliques_temp_add_22 = fc.filter_candidates(cliques_temp_add_22, flag=True)

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

    timenow = datetime.datetime.now()
    print('Tiempo Total:', timenow - time_all)
    print('termina alineamiento de cliques, se procede alineamiento de residuo a residuo')

    # AQUI COMIENZA EL ALINEAMIENTO RESIDUO A RESIDUO
    pc = parejas_cliques_finales
    pc = fc.filter_candidates(pc)

    cliques_finales = []
    for cand_1, cand_2, mat_rot in pc:
        cliques_finales.append([cand_1, cand_2])

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
        # aveces truena si los pdbs no tienen los numeros de residuos continuos.
        coord_new_1 = [[res.GetAtom('CA').coord for res in res_conclq_1 if i[0] == res.resx] for i in parejas]
        coord_new_2 = [[res.GetAtom('CA').coord for res in res_conclq_2 if i[1] == res.resx] for i in parejas]

        coord_new_1 = np.array([y for x in coord_new_1 for y in x], dtype=np.float)
        coord_new_2 = np.array([y for x in coord_new_2 for y in x], dtype=np.float)

        bari_new_1 = coord_new_1.mean(0)
        bari_new_2 = coord_new_2.mean(0)

        vecs_center_1 = coord_new_1 - bari_new_1
        vecs_center_2 = coord_new_2 - bari_new_2

        # Se genera matriz de rotacion con vectores centricos de parejas anteriores
        matriz_R = fc.matrix_R(vecs_center_1, vecs_center_2)
        matriz_rotacion = fc.rotation_matrix(matriz_R)
        # se aplica matriz de rotacion a coordenadas de proteina a rotar
        # la proteina consiste en solo carbonos alfa
        vector_rotado = fc.rotation_vectors(vecs_center_cnclq_1, matriz_rotacion)

        protein_trasladado_rotado = vector_rotado + bari_new_2

        protein_to_compare = coord_conclq_2

        return (protein_trasladado_rotado, protein_to_compare,matriz_rotacion, bari_new_2)


    # cosas que tengo que declarar para que no me diga nada pycharm...
    matriz_rotacion = []
    bari_new_2 = []
    winner_parejas = []

    for cand_1, cand_2, mat_rot in pc:

        # primera iteracion sin cliques se aplica la matriz de rotacion y baricentro
        # print('***********************************************************')
        # print(val, cand_1, cand_2)
        res_sinclq_1 = [res for res in pdb11 if res.resx not in cand_1]
        res_sinclq_2 = [res for res in pdb22 if res.resx not in cand_2]

        coord_sinclq_1 = np.array([res.GetAtom('CA').coord for res in res_sinclq_1], dtype=np.float)
        coord_sinclq_2 = np.array([res.GetAtom('CA').coord for res in res_sinclq_2], dtype=np.float)

        bari_1 = coord_sinclq_1.mean(0)
        bari_2 = coord_sinclq_2.mean(0)

        vecs_center_1 = coord_sinclq_1 - bari_1
        # aplico matriz de rotacion de cliques a vectores centricos sin clique
        vector_rotado = fc.rotation_vectors(vecs_center_1, mat_rot)
        protein_trasladado_rotado = vector_rotado + bari_2

        protein_to_compare = coord_sinclq_2

        # apilo la distancia y la pareja de residuos correspondientes si cumple con que el RMSD sea menor a 3.5
        residuos_match = fc.fun_resiudos_match(protein_trasladado_rotado, protein_to_compare,
                                            res_sinclq_1, res_sinclq_2)
        # filtro parejas
        cand_n = fc.filter_pairs(residuos_match, flag=False)
        # calculo el SO
        so_temp = round(len(cand_n) / (number_of_residues_final - 7), 4)
        # print('PRE_SO:', so_temp)
        so_temp_plus_1 = 0.0

        # Refinamiento por medio de las parejas seleccionadas y el clique.
        while so_temp_plus_1 < so_temp:  # Primer refinamiento
            parejas = [i[1] for i in cand_n]
            for i, j in zip(cand_1, cand_2):
                parejas.insert(0, (i, j))
            # print(parejas)
            # aqui comienza el segundo alineamiento!! Refinamiento
            ptr, ptc, mr, bc = gen_rot_matrix_ref(parejas)
            # match residuos ordenado por distancia
            rm = fc.fun_resiudos_match(ptr, ptc, res_conclq_1, res_conclq_2)

            # quitar residuos repetidos
            cand_n = fc.filter_pairs(rm, flag=False)
            so_temp_plus_1 = round(len(cand_n) / number_of_residues_final, 4)
            so_temp_minus_1 = so_temp
            # print(so_temp_plus_1)
            if so_temp_plus_1 < so_temp:  # evita infinite loop
                break

            # print(so_temp_minus_1, so_temp_plus_1)

            # Rerefinamiento por si puede ir encontrando nuevas y mejores parejas
            while so_temp_minus_1 < so_temp_plus_1:  # segundo refinamiento iterativo
                so_temp_minus_1 = so_temp_plus_1
                parejas = [i[1] for i in cand_n]
                for i, j in zip(cand_1, cand_2):
                    parejas.insert(0, (i, j))

                # print(parejas)
                # aqui comienza el segundo alineamiento!! Refinamiento
                ptr, ptc, matriz_rotacion, bari_new_2 = gen_rot_matrix_ref(parejas)
                # match residuos ordenado por distancia
                rm = fc.fun_resiudos_match(ptr, ptc, res_conclq_1, res_conclq_2)

                # quitar residuos repetidos
                cand_n = fc.filter_pairs(rm, flag=False)
                so_temp_plus_1 = round(len(cand_n) / number_of_residues_final, 4)

                # print(so_temp_minus_1, so_temp_plus_1)

            # actualizacion de datos
            if so_temp_plus_1 < so_temp_minus_1:
                so_temp_plus_1 = so_temp_minus_1
            # actualizacion de datos

            if so_temp_plus_1 > so_temp:
                so_temp = so_temp_plus_1

        # check que si este guardando el SO
        # print(so_winner)

        # Si supera el SO ganador se guardan los parametros y se actualiza el SO
        if so_temp > so_winner:

            so_winner = so_temp  # actualizacion so
            winner_matrix_rotation = matriz_rotacion  # actualizacion mr
            winner_baricenter = bari_new_2  # actualizacion bc
            candidatos = [cand_1, cand_2]  # actualizacion de parejas de cliques estrella
            winner_parejas = cand_n   # actualizacion de parejas y distancia.
            print('========================='*3)
            print('viejo so:',so_temp_plus_1)
            print('cliques', candidatos)
            print('numero de parejas', len(cand_n))
            print('iteracion %s' % val, 'SO: %1.4f' % so_temp)
            print('RMSD:', np.mean([x[0] for x in cand_n]))
            # print('parejas:', [x[1] for x in cand_n])
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

    # se escribe el nuevo pdb rotado y trasladado
    print('escribiendo resultados')

    num_match_atoms = len(winner_parejas)
    num_res_total = number_of_residues_final
    rmsd = round(np.mean([x[0] for x in winner_parejas]), 4)
    so_out = so_winner
    parejas = sorted([x[1] for x in winner_parejas])

    df = pd.DataFrame([pdb1.name+'_'+pdb2.name, 'align_s', candidatos, num_match_atoms, num_res_total,
                        so_out, rmsd, parejas],
                        index=['proteinaA_proteinaB', 'grupo', 'cliques_ganadores', 'num_parejas', 'num_residuos_total',
                        'SO', 'RMSD', 'parejas']).T

    df.to_csv(directory+pdb1.name+'_'+pdb2.name+'.csv')


    # Actualizacion de coordendas
    file_output_name = directory+pdb1.name+'_'+pdb2.name+'_'+str(datetime.datetime.now())[:19]

    coord_protein_1 = np.array([res.GetAtom(name).coord for res in pdb11 for name in res.atomnames],
                               dtype=np.float)
    bari_full_1 = coord_protein_1.mean(0)
    vecs_center_protein_1 = coord_protein_1 - bari_full_1

    # aplicacion de la rotacion y traslacion a toda la proteina
    vector_rotado = fc.rotation_vectors(vecs_center_protein_1, winner_matrix_rotation)
    protein_trasladado_rotado = vector_rotado + winner_baricenter

    # actualizacion de coordenadas
    k = 0
    for res in pdb11:
        for atom in res.atomnames:
            setattr(res.GetAtom(atom), 'coord', protein_trasladado_rotado[k])
            k = k+1

    # escritura del pdb
    pdb1.WriteToFile(file_out_name=file_output_name)

    #tiempo de ejecucion
    timenow = datetime.datetime.now()
    print('el cutoff dihedral era de:', global_cutoff)
    print('Tiempo Total:', timenow - time_all)
    print('termine puedes alinear utilizando los pdbs %s %s y el alineamiento %s' % (file1[-10:-4], file2[-10:-4],
                                                                                     file_output_name))

