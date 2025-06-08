import os
import subprocess

# Define the base command
base_command = "python src/models/autoCOT-o4-mini/autoCOTo4.py"

# Define the base directory for experiments
experiments_base_dir = "/Users/kevinawang/Documents/GitHub/VA-classifier/experiments"

# Define the flag combinations
flag_combinations = [
    "--summary",
    "--summary --without_va",
    "--length_pruned",
    "--length_pruned --without_va",
    "--va_pruned",
    "--va_pruned --without_va",
    "",
    "--without_va"  # Added baseline without VA
]

# Define the n values
n_values = [2, 4, 6, 8]

# Run all combinations sequentially
for flags in flag_combinations:
    # Determine VA usage
    va_usage = "without_VA" if "--without_va" in flags else "with_VA"
    
    # Determine flag type
    if "--summary" in flags:
        flag_type = "summary"
    elif "--length_pruned" in flags:
        flag_type = "length_pruned"
    elif "--va_pruned" in flags:
        flag_type = "va_pruned"
    else:
        flag_type = "baseline"

    for n in n_values:
        # Construct the directory path
        output_dir = os.path.join(experiments_base_dir, va_usage, flag_type, f"n{n}")
        os.makedirs(output_dir, exist_ok=True)

        # Construct the command
        log_filename = os.path.join(output_dir, f"nohup_{flags.replace(' ', '_')}_n{n}.out")
        command = f"nohup {base_command} {flags} --n {n} > {log_filename} 2>&1"
        print(f"Running: {command}")
        # Execute the command sequentially
        subprocess.run(command, shell=True, check=True) 