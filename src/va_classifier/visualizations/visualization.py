"""
Temporal VA Step-wise Progression Visualization (§5.2 & §6 Discussion)

Purpose: Creates step-wise line plots showing valence and arousal progression
         over time during clinical interviews, stratified by gender and depression
         status. Provides temporal analysis of emotional dynamics through interviews.

Paper Section: §5.2 PHQ-8 Results & §6 Discussion (temporal progression analysis)
Application: Linear temporal analysis of emotional states during clinical interviews

This script generates step-wise temporal progression visualizations:

1. Demographic Group Analysis:
   - Male PHQ Binary 0: Non-depressed male participants
   - Male PHQ Binary 1: Depressed male participants
   - Female PHQ Binary 0: Non-depressed female participants
   - Female PHQ Binary 1: Depressed female participants
   - Average PHQ Binary 0: Combined non-depressed population
   - Average PHQ Binary 1: Combined depressed population

2. Temporal Progression Features:
   - X-axis: Interview step/utterance number (temporal sequence)
   - Y-axis: Mean VA scores (-1 to 1 range)
   - Dual line plots: Separate lines for valence and arousal
   - Group-averaged trajectories showing population-level patterns

3. Visualization Design:
   - Line plot with markers: Circle markers for valence, square markers for arousal
   - Contrasting line styles: Solid line for valence, dashed line for arousal
   - Step-by-step progression showing temporal emotional evolution
   - Grid overlay for precise value reading

4. Clinical Interpretation:
   - Valence trends: Emotional positivity/negativity over interview time
   - Arousal trends: Emotional activation/deactivation over interview time
   - Temporal patterns: How emotions systematically change during interviews
   - Group differences: Comparative emotional trajectories across populations

Research Questions Addressed:
- Do valence and arousal follow different temporal patterns during interviews?
- Are there systematic emotional changes from interview beginning to end?
- Do depressed individuals show distinct temporal emotional patterns?
- How do gender differences manifest in temporal VA progression?

Expected Clinical Patterns:
- Depressed groups: Consistently lower valence throughout interviews
- Arousal patterns: Potential differences in emotional activation over time
- Temporal stability: Some groups may show more stable vs variable patterns
- Gender effects: Possible differences in emotional trajectory shapes

Technical Implementation:
- Pandas-based temporal data alignment across participants
- NumPy array operations for efficient step-wise aggregation
- Matplotlib line plotting with dual-series visualization
- Automatic file discovery and robust missing data handling

Data Processing Pipeline:
1. Load participant demographics and PHQ-8 binary classifications
2. Discover and load individual transcript files with VA scores
3. Group participants by gender and depression status
4. Align temporal sequences across participants (padding shorter interviews)
5. Calculate group-averaged valence and arousal at each time step
6. Generate dual-line temporal progression plots

Output Files:
- Male_PHQ_Binary_0_steps.png: Non-depressed male temporal progression
- Male_PHQ_Binary_1_steps.png: Depressed male temporal progression
- Female_PHQ_Binary_0_steps.png: Non-depressed female temporal progression
- Female_PHQ_Binary_1_steps.png: Depressed female temporal progression
- Average_PHQ_Binary_0_steps.png: Combined non-depressed progression
- Average_PHQ_Binary_1_steps.png: Combined depressed progression

Visualization Advantages:
- Clear temporal sequence visualization (unlike 2D trajectory plots)
- Separate valence and arousal trend analysis
- Precise step-by-step emotional progression tracking
- Easy identification of temporal patterns and inflection points

Research Applications:
- Temporal validation of VA-based depression classification
- Identification of critical interview moments with emotional changes
- Development of temporally-aware diagnostic algorithms
- Analysis of interview protocol effects on emotional state

Clinical Significance:
- Reveals optimal interview timing for emotional assessment
- Identifies temporal windows with strongest diagnostic signals
- Supports development of adaptive interview protocols
- Enables real-time emotional monitoring during clinical sessions

Related Files:
- va_classifier.visualizations.visualization_2d — 2D VA trajectory plots
- va_classifier.visualizations.metrics — Quantitative temporal analysis
- va_classifier.visualizations.create_grid — Demographic comparison grids
- test_split.csv — Participant demographics and labels

Usage:
    python visualization.py
    # Requires: test_split.csv and *_Transcript.csv files in working directory
    # Outputs: Step-wise progression plots in step_plots/ directory

Note: These step-wise visualizations complement 2D trajectory plots by providing
      linear temporal analysis, making it easier to identify specific time points
      where emotional patterns diverge between clinical populations.
"""

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

if not os.path.exists('test_split.csv'):
    raise FileNotFoundError("Could not find 'test_split.csv' in the current folder.")
labels = pd.read_csv('test_split.csv')

transcript_files = glob.glob('*_Transcript.csv')
if len(transcript_files) == 0:
    raise FileNotFoundError("No '*_Transcript.csv' files found in the current folder.")

transcripts = {}
for fp in transcript_files:
    pid = int(os.path.basename(fp).split('_')[0])
    df = pd.read_csv(fp).reset_index().rename(columns={'index': 'Step'})
    transcripts[pid] = df

groups = {
    'Male PHQ Binary 0':    labels[(labels['Gender']=='male')   & (labels['PHQ_Binary']==0)]['Participant_ID'].tolist(),
    'Male PHQ Binary 1':    labels[(labels['Gender']=='male')   & (labels['PHQ_Binary']==1)]['Participant_ID'].tolist(),
    'Female PHQ Binary 0':  labels[(labels['Gender']=='female') & (labels['PHQ_Binary']==0)]['Participant_ID'].tolist(),
    'Female PHQ Binary 1':  labels[(labels['Gender']=='female') & (labels['PHQ_Binary']==1)]['Participant_ID'].tolist(),
    'Average PHQ Binary 1': labels[labels['PHQ_Binary']==1]['Participant_ID'].tolist(),
    'Average PHQ Binary 0': labels[labels['PHQ_Binary']==0]['Participant_ID'].tolist(),
}

output_folder = 'step_plots'
os.makedirs(output_folder, exist_ok=True)

for group_name, id_list in groups.items():
    valid_ids = [pid for pid in id_list if pid in transcripts]
    if not valid_ids:
        print(f"Skipping '{group_name}' (no matching transcripts).")
        continue

    max_steps = max(transcripts[pid].shape[0] for pid in valid_ids)

    val_df = pd.DataFrame(index=range(max_steps))
    aro_df = pd.DataFrame(index=range(max_steps))

    for pid in valid_ids:
        df = transcripts[pid]
        v = df['valence'].reset_index(drop=True).reindex(range(max_steps))
        a = df['arousal'].reset_index(drop=True).reindex(range(max_steps))
        val_df[pid] = v
        aro_df[pid] = a

    mean_valence = val_df.mean(axis=1)
    mean_arousal = aro_df.mean(axis=1)

    steps = np.arange(max_steps)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, mean_valence.values, marker='o', linestyle='-', label='Valence', linewidth=2)
    ax.plot(steps, mean_arousal.values, marker='s', linestyle='--', label='Arousal', linewidth=2)

    ax.set_title(group_name, fontsize=14)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    filename = group_name.replace(' ', '_') + '_steps.png'
    filepath = os.path.join(output_folder, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved '{group_name}' step‐based plot as '{filepath}'")
