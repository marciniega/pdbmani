# concatenar resultados
import pandas as pd
# leer documentos
import glob
import os
import sys

print(sys.argv[1])
os.chdir(sys.argv[1])

df_tot = pd.DataFrame()
for file in glob.glob("*.clique"):
    print(file[:13])
    # get data
    data = open(file, 'r').readlines()[:6]
    # numero a comparar
    num_match_atoms = data[0].split()[-1]
    rmsd = data[1].split()[-1]
    so = data[2].split()[-1]
    num_res_total = data[3].split()[11]

    df = pd.DataFrame([file[:13], 'click', [], num_match_atoms, num_res_total, so, rmsd, []],
                 index=['proteinaA_proteinaB', 'grupo', 'cliques_ganadores', 'num_parejas', 'num_residuos_total',
                        'SO', 'RMSD', 'parejas'],
                      ).T

    df_tot = pd.concat([df_tot, df], axis=0)
    df_tot.to_csv(file[:13]+'_click.csv')

print(df_tot)