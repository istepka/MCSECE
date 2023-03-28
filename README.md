# Multi-criteria approach for selecting an explanation from the set of counterfactuals produced by an ensemble of explainers


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
    Now under `./experiments` you should see a folder with your current date.  
    This might take some time. In order to speed things up you can easily run this script in parallel for each dataset.

1. (optional) After data is properly generated experiments from the paper can be obtained by running:  
    For Tables 2,3,4,5 and Figure 2
    ```bash
    python3 experiments/experiment1.py
    ```

    For Table 6
    ```bash
    python3 experiments/experiment1_stats.py
    ```


***
