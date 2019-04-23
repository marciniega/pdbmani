import pandas as pd
import ast


def get_pairs_click(path):
    file = open(path)
    lista = []
    for i in file.readlines()[6:]:
        lista.append(i.split("\n")[0].split())

    df = pd.DataFrame(lista, columns=['Chain_protA', 'ResNum_protA', 'ResName_protA', 'Atom_protA',
                                        'Chain_protB', 'ResNum_protB', 'ResName_protB', 'Atom_protB'])
    df.drop(0, inplace=True)
    par = []
    for i in df.index:
        par.append(ast.literal_eval('('+df.loc[i].ResNum_protA+','+df.loc[i].ResNum_protB+')'))

    df['parejas'] = par
    return df

