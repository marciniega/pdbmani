#!/usr/bin/env python
import string
from sys import argv,stderr,stdout
from os import popen,system
from os.path import exists

dictionary = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D', \
              'CYS':'C','GLU':'E','GLN':'Q','GLY':'G', \
              'HIS':'H','ILE':'I','LEU':'L','LYS':'K', \
              'MET':'M','PHE':'F','PRO':'P','SER':'S', \
              'THR':'T','TRP':'W','TYR':'Y','VAL':'V',
              'HIE':'H','HID':'H','HIP':'H'}

assert( len(argv)>1)
pdbname = argv[1]

lines = open(pdbname,'r').readlines()

oldresnum = '   '
count = 1;

outid = stdout

outid.write('\n      Atention\n \
             This is a preliminar version. Gaps in the structure are not suported.\n\n')
for line in lines:
        line_edit = line
        if line[0:3] == 'TER':
           outid.write('\n')
           outid.write('New chain\n')
           count = 1
           continue

        if line_edit[0:4] == 'ATOM':
                if 'N  ' == line_edit[13:16]:
                    count = count + 1
                    resn = line[17:20]
                    try:
                      outid.write(dictionary[resn])
                      if count==61:
                         outid.write('\n')
                         count = 1
                    except:
                      pass
outid.write('\n')
outid.close()
