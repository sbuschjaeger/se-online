# Shrub Ensembles For Online Classification

This repsitory contains to code for our ICDM 2021 submission "Shrub Ensembles for Online Classification". In order to run the experiments you first need to download the corresponding data-sets. In each folder of each dataset you can find a `init.sh` script which downloads and prepares the data. For example to run experiments on the `gas-sensor` dataset just do

    cd gas-sensor
    ./init.sh

If you want to download all dataset you can simply call `init_all.sh`.

To run the experiments several python packages are required. If you manage your python packages via Anaconda, then you can simply create the provided conda environment via

    conda env create -f environment.yml
    conda activate se-online

Note, that the conda enviroment is named `se-online` which might conflict with your existing environments. In that case, feel free to change the name in the `environment.yml` file directly. Once the environment is created, you need to installed two additional packages:

    pip install -e experiment_runner
    pip install -e ShrubEnsembles

The `experiment_runner` package is used to run the experiments, whereas the `ShrubEnsemble` package contains the actual implementation of the Shrub Ensemble algorithm. Note that Shrub Ensembles provides a python and a c++ backend and therefore requires a c++ compiler and cmake. The conda enviroment should automatically install all necessary packages and compile the source code on the fly. However, if for some reason you are not able to compile the code, you can try to utilize the python backend. In this case however, you'll have to manually remove the CMake files form `ShrubEnsembles/setup.py` and adapt imports in the sources files accordingly. 

Central for running the experiments in the `run.py` file. It takes several arguments: 

- "-j"/"--n_jobs": Number of jobs for multiprocessing pool. Default is `1` 
- "-d", "--dataset": "Datasets used for experiments. This can be a list (space separated) for multiple datasets. Default is `gas-sensor`
- "-c" / "--n_configs": Number of hyperparameter configurations per learner. Default is `1`.
- "-t" / "--timeout": Maximum number of seconds per hyperparameter configuration on a dataset. If the runtime exceeds the provided value, stop execution of that single configuration. Defauls is `3600` seconds. 
- "-o" / "--out_path": "Path where results should be stored. Default is the current folder. 

If you set `n_jobs` to `1` you will get detailed information about the current running algorithm such as its test-then-train accuracy, model size etc. If you set `n_jobs` greater than `1` a simple progress bar is displayed. 

An example call which checks 50 different hyperparameter configurations using 50 threads would be:
    ./run.py -d gas-sensor -c 50 -j 50 -t 3600 -o /some/nice/place

In order to view the results and generate the tables / plots as presented in the paper you can use the `explore_online.py` file. I recommend to use Visual Studio Code or a similar tool which supports the exection of python code snippest / cells. The file is fairly well documented to please refer to it directly.  