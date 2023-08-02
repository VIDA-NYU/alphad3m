#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 16GB
#SBATCH --time 01:10:00
#SBATCH --job-name=automl_job
#SBATCH --output logs/automl_job.out
#SBATCH --mail-user=rl3725@nyu.edu

./sing << EOF

python automlbenchmark/runbenchmark.py $1 $2 1h4c -f 0 -u user_config/

EOF