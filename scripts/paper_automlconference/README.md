# Paper Experiments


## OpenML Experiments

We used the AutoML Benchmark (AMLB) to run these experiments locally using Python 3.8. To reproduce these experiments, 
follow the next steps.

1. Go to the *openml_experiments* folder and create the *openml_datasets* folder (AMLB will download the datasets within 
this folder).
```
cd openml_experiments
mkdir openml_datasets
```

2. Clone and install AMLB, version v2.0.6. See the [AMLB repository](https://github.com/openml/automlbenchmark/tree/v2.0.6/) 
for additional details about the installation. 

```
git clone https://github.com/openml/automlbenchmark.git --branch v2.0.6 --depth 1
cd automlbenchmark
pip install -r requirements.txt
```

4. To test the installation, run the following command. You should get valid ML pipelines after running it.
```
python automlbenchmark/runbenchmark.py AlphaD3M  openml/t/3945  -f 0 -u user_config/ -i openml_datasets/
```

5. We ran all the systems (AutoWEKA, TPOT, H2O, AutoGluon, Auto-Sklearn, and AlphaD3M) using Singularity containers in 
SLURM batch jobs in the [NYU Greene Cluster](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene). To run the 
experiments in this cluster, run `./run_all_automlbenchmark.sh` otherwise run `./run_all_automlbenchmark_no_sbatch.sh`. 
All the results will be stored in the `./results/results.csv` file.


6. To generate all the tables and figures reported in the paper for these experiments, run the Jupyter Notebook 
`d3m_datasets_experiments.ipynb`.


## D3M Experiments

The experiments using the D3M datasets cannot be fully reproduced since they were conducted by a third party and some 
of the systems are not open-source.  However, we provide all the scripts to calculate the metrics and numbers reported 
for the D3M datasets. 

Go to the `d3m_experiments` folder and run the Jupyter Notebook `d3m_datasets_experiments.ipynb`. It also contains the 
scripts to generate the tables and figures reported in the paper for these experiments.