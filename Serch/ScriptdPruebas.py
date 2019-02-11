import read_pdb_tools as rpt

file1 = 'pdbs/1xxa.pdb'
pdb1 = rpt.PdbStruct(file1)
pdb1.AddPdbData("%s" % file1)
pdb1.WriteToFile(file_out_name='algo.data', flag_trj=False)