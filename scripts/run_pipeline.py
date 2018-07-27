#!/usr/bin/env python3
import json
import os
import sys
import uuid

from d3m_ta2_nyu.ta2 import D3mTa2


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s <config> <pipeline_uuid>\n' % sys.argv[0])
        sys.exit(1)

    with open(sys.argv[1]) as config_file:
        config = json.load(config_file)
    storage = config['temp_storage_root']

    ta2 = D3mTa2(storage_root=storage,
                 pipelines_considered=os.path.join(storage, 'pipelines_considered'),
                 executables_root=os.path.join(storage, 'executables'))

    result = ta2.run_pipeline(uuid.UUID(hex=sys.argv[2]),
                              config['training_data_root'],
                              config['problem_root'])
    print(result)
