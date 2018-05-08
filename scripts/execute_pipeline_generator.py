import os
from d3m_ta2_nyu.alphad3m_edit import PipelineGenerator
import pickle
import sys

dataset_index = int(sys.argv[1])

f=open('/home/yk38/ta2_branch/alphad3m_edit/scripts/dataset_names.txt')
dataset_names = [line.strip() for line in f.readlines()]


for dataset in dataset_names:
    if os.path.isfile(os.path.join('/home/yk38/metafeatures', dataset+'_metafeatures.pkl')):
        PipelineGenerator.main(dataset)
    else:
        print('no metafeatures')
