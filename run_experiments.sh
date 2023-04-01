#!/bin/bash

dataset_name="adult"

for i in {0..250}
do
    echo "Running experiment $i"
    # Activate conda environment
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate ecemosp
    # cd ./src
    # Run experiment
    python3 src/run_experiment.py $dataset_name $i
done