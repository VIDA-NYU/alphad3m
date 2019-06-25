import os
from d3m_ta2_nyu.alphad3m import interface_alphaautoml
import pickle
import sys

dataset_index = int(sys.argv[1])

f=open('/home/yk38/ta2_branch/alphad3m/scripts/dataset_names.txt')
dataset_names = [line.strip() for line in f.readlines()]


for dataset in dataset_names:
    if os.path.isfile(os.path.join('/home/yk38/metafeatures', dataset+'_metafeatures.pkl')):
        interface_alphaautoml.main(dataset)
    else:
        print('no metafeatures')
