# Multi-criteria approach for selecting an explanation from the set of counterfactuals produced by an ensemble of explainers

This is a repository for the paper: "Multi-criteria approach for selecting an explanation from the set of counterfactuals produced by an ensemble of explainers". 

The paper is available at: <...>

## 1. Installation
```
git clone --recurse-submodules https://github.com/iggyyy/ecemosp
conda create -n <env_name> python==3.10
pip install -r requirements.txt
```
Go to `./modules` and check if CARLA and CFEC modules are loaded there. If not get them using git submodule command.

***
## 2. Reproducing paper experiments
To reproduce paper experiments (tested on Ubuntu 20) follow the steps:

1. Set your conda environment path in `run_reproducibility.sh` script. 
1. Generate explanations for the test datasets
    ```bash
    > chmod u+x run_reproducibility.sh
    > ./run_reproducibility.sh
    ```
    Now under `./experiments/data` you should see a folder with your current date.  
    This might take some time. In order to speed things up you can easily run this script in parallel for each dataset.

1. (optional) After data is properly generated experiments from the paper can be obtained by running:  
    For Tables 2,3,4,5 (scores) and Figure 2 (barycentric plots)
    ```bash
    python3 experiments/experiment1.py
    ```

    For Table 6 and 7 (statistics)
    ```bash
    python3 experiments/experiment1_stats.py
    ```

    For radar plots
    ```bash
    python3 experiments/radar_graphs.py
    ```

    For pareto front plots
    ```bash
    python3 experiments/pareto_front_graphs.py
    ```


***
