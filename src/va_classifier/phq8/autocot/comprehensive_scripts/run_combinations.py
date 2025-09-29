"""
AutoCoT Parallel Experimental Runner (§3.2.2 & §4)

Purpose: Parallel execution of all AutoCoT experimental configurations for PHQ-8 prediction
         to generate comprehensive results across VA context, pruning strategies, and demo counts.
         This is the parallel version of combinations.py for faster experimental execution.

Paper Sections: §3.2.2 AutoCoT Few-Shot Prompting & §4 Cross-cutting Experiments
Experimental Design: Full factorial design testing 8 conditions × 4 demo counts = 32 experiments

This script automates the parallel execution of all AutoCoT experimental combinations:

Experimental Conditions:
1. Baseline (full transcripts)
2. Summary (condensed transcripts) 
3. Length Pruned (remove excessive content)
4. VA Pruned (emotion-based content selection)

Each condition tested with/without VA context:
- With VA: Valence-arousal scores integrated into prompts
- Without VA: Text-only baseline for comparison

Demo Count Optimization:
- n=2, 4, 6, 8 demonstration examples per condition
- Tests impact of few-shot example quantity on performance

Key Difference from combinations.py:
- PARALLEL EXECUTION: Uses `nohup ... &` to run experiments concurrently
- FASTER: Reduces total runtime from ~8-12 hours to ~2-3 hours
- RESOURCE INTENSIVE: Requires careful monitoring of API rate limits and system resources

Output Structure:
experiments/
├── with_VA/
│   ├── baseline/     # Full transcripts + VA
│   ├── summary/      # Summarized + VA  
│   ├── length_pruned/ # Length pruned + VA
│   └── va_pruned/    # VA pruned + VA
└── without_VA/
    ├── baseline/     # Full transcripts only
    ├── summary/      # Summarized only
    ├── length_pruned/ # Length pruned only
    └── va_pruned/    # VA pruned only

Each subdirectory contains n2/, n4/, n6/, n8/ folders with experimental logs.

Research Questions Addressed:
- Does VA context improve AutoCoT performance across all conditions?
- Which transcript preprocessing strategy yields best results?
- What is the optimal number of demonstration examples?
- How do pruning strategies interact with VA enhancement?

Usage:
    python run_combinations.py
    # Runs all 32 experimental combinations in parallel
    # Each experiment logged to: experiments/{va_usage}/{condition}/n{n}/nohup_*.out
    
    # Monitor running processes:
    ps aux | grep autoCOTo4
    
    # Check logs in real-time:
    tail -f experiments/with_VA/baseline/n8/nohup__n8.out

Cautions:
- Monitor OpenAI API rate limits carefully
- Ensure sufficient system resources (CPU, memory)
- Consider staggered execution if rate limits are hit
- Use combinations.py for sequential execution if needed

Results:
- Comprehensive experimental logs for paper Tables and Figures
- Systematic comparison across all AutoCoT configurations
- Foundation for statistical analysis of VA impact and pruning effectiveness

Related Files:
- va_classifier.phq8.autocot.o4_mini — Core AutoCoT implementation
- combinations.py — Sequential version (safer for rate limits)
- results/phq8/autocot/ — Organized experimental results

Note: This parallel approach trades execution time for resource complexity.
      Monitor system performance and API usage during execution.
"""

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

# Run all combinations
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
        command = f"nohup {base_command} {flags} --n {n} > {log_filename} 2>&1 &"
        print(f"Running: {command}")
        # Execute the command
        subprocess.run(command, shell=True, check=True) 