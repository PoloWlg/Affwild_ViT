import itertools
import os

sh_path = "shells_grid_search_proposed2_bert"
if not os.path.exists(sh_path):
    os.makedirs(sh_path)
# Grid search parameters
learning_rates = [1e-2, 1e-3, 1e-4]  # Learning rates for grid search
batch_sizes = [8,16,32,64]  # Adjusted batch sizes for grid search
dropout = [0]  # Dropout rate for the body model


# Define available GPUs
gpu_ids = [0, 1, 2, 3]

root_dir = "/home/ens/AS84330/Context/ABAW3_EXPR4"
# Create all combinations of hyperparameters
grid = list(itertools.product(learning_rates, batch_sizes, dropout))

# Split jobs evenly across GPUs
split_jobs = [[] for _ in gpu_ids]
for i, config in enumerate(grid):
    gpu_index = i % len(gpu_ids)
    split_jobs[gpu_index].append(config)

# Output shell scripts for each GPU
for gpu_index, jobs in enumerate(split_jobs):
    script_name = f"{sh_path}/gpu_{gpu_index}.sh"
    with open(script_name, "w") as f:
        f.write("#!/bin/bash\n\n")
        # f.write(f"export CUDA_VISIBLE_DEVICES={gpu_index}\n\n")
        for lr, bs, dropout in jobs:
            cmd = f"python {root_dir}/main.py \
                -gpu {gpu_index} \
                -learning_rate {lr} \
                -batch_size {bs} \
                -fusion_method Video_only \
                -model_name Video_only \
                -experiment_name proposed2/proposed2_lr_{lr}_bs_{bs} \
                -modality EXPR_continuous_label context video \
                -num_epochs 15 \
                -context_feature_model bert \
                "
            f.write(f"{cmd}\n")
    os.chmod(script_name, 0o755)  # Make script executable
    
# Create sh file that execute all scripts
with open(f"{sh_path}/run_all.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    for i in range(len(gpu_ids)):
        f.write(f"./gpu_{i}.sh &\n")
    f.write("wait\n")



print("✅ Shell scripts created: gpu_0.sh, gpu_1.sh, gpu_2.sh, gpu_3.sh")