import os
from d3m_ta2_nyu.alphad3m_edit import PipelineGenerator
import pickle
import sys

dataset_index = int(sys.argv[1])

f=open('/Users/yamuna/D3M/ta2_branches/alphad3m_edit/scripts/dataset_names.txt')
dataset_names = [line.strip() for line in f.readlines()]

dataset = dataset_names[dataset_index]
dataset = 'LL0_21_car'
if os.path.isfile(os.path.join('/Users/yamuna/D3M/data/metafeatures', dataset+'_metafeatures.pkl')):
    PipelineGenerator.main(dataset)
else:
    print('no metafeatures')
