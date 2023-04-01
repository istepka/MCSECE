#!/bin/bash

# -- PARAMETERS TO SET --
conda_path=~/anaconda3/etc/profile.d/conda.sh # Change this to your conda path if needed
conda_env_name="<YOUR_CONDA_ENV_NAME>"
# -- PARAMETERS TO SET --

# -- EXPERIMENT PARAMETERS --
dataset_names=("german" "compas" "adult" "fico")
dataset_test_lengths=(100 250 250 250)
# -- EXPERIMENT PARAMETERS --

# -- RUN EXPERIMENTS--
idx=0
for dataset_name in ${dataset_names[@]}
do
    dataset_test_length=${dataset_test_lengths[$idx]}
    for i in $(seq 0 $dataset_test_length)
    do
        echo "Running experiment $i for dataset $dataset_name out of $dataset_test_length"
        source $conda_path
        conda activate $conda_env_name
        python3 src/run_experiment.py $dataset_name $i
    done
    idx=$((idx+1))
done
# -- RUN EXPERIMENTS --


# -- RUN GENERATING SUMMARIES AND PLOTS --
# This section should be executed only after all the experiments have been run
# If previous section is run in parallel this section should be commented and 
# ran after all the experiments have been done

# Generate experiment results (Multi criteria tables and barycentric plots)
python3 experiments/experiment1.py

# Generate experiment stats (Tables with stats for explainer on each dataset)
python3 experiments/experiment1_stats.py

# Generate experiment results (barycentric plots)
python3 experiments/radar_graphs.py

# Generate experiment results (pareto front plots)
python3 experiments/pareto_front_graphs.py

# -- RUN GENERATING SUMMARIES AND PLOTS --