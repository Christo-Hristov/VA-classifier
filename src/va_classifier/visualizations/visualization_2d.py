"""
2D VA Trajectory Visualization by Demographics (§5.2 & §6 Discussion)

Purpose: Creates 2D valence-arousal trajectory plots showing temporal emotional
         progressions during clinical interviews, stratified by gender and depression
         status. Visualizes emotional journeys in the VA coordinate system.

Paper Section: §5.2 PHQ-8 Results & §6 Discussion (temporal trajectory analysis)
Application: Visual analysis of emotional dynamics during clinical interviews

This script generates publication-ready 2D VA trajectory visualizations:

1. Demographic Group Analysis:
   - Male PHQ Binary 0: Non-depressed male participants
   - Male PHQ Binary 1: Depressed male participants
   - Female PHQ Binary 0: Non-depressed female participants
   - Female PHQ Binary 1: Depressed female participants
   - Average PHQ Binary 0: Combined non-depressed population
   - Average PHQ Binary 1: Combined depressed population

2. VA Coordinate System Visualization:
   - X-axis: Mean valence (-0.5 to 0.5, negative=unpleasant, positive=pleasant)
   - Y-axis: Mean arousal (-0.5 to 0.5, negative=calm, positive=activated)
   - Quadrant labels: Happiness, Anger, Sadness, Tenderness
   - Temporal progression: Color-coded trajectory from interview start to end

3. Trajectory Features:
   - Connected line plot showing temporal progression
   - Color-coded scatter points (red=start, blue=end)
   - Step-by-step emotional journey visualization
   - Group-averaged trajectories for population-level insights

4. Clinical Interpretation Quadrants:
   - Top-right (High V, High A): Happiness, excitement, positive activation
   - Top-left (Low V, High A): Anger, frustration, negative activation
   - Bottom-left (Low V, Low A): Sadness, depression, negative deactivation
   - Bottom-right (High V, Low A): Tenderness, calm satisfaction, positive deactivation

Research Questions Visualized:
- Do depressed individuals follow different emotional trajectories during interviews?
- Are there systematic patterns in how emotions evolve from interview start to end?
- Do gender differences manifest in VA trajectory patterns?
- Which VA quadrants are most characteristic of different clinical populations?

Visualization Design:
- Centered coordinate system with axes through origin
- Color gradient from red (early) to blue (late) showing temporal progression
- Quadrant emotion labels for clinical interpretation
- Standardized axis ranges (-0.5 to 0.5) for cross-group comparison
- High-resolution output (300 DPI) for publication quality

Expected Clinical Patterns:
- Depressed groups: Trajectories concentrated in negative valence regions
- Non-depressed groups: More positive valence, potentially greater variability
- Gender differences: Possible variations in arousal patterns and trajectory shapes
- Temporal effects: Systematic movement patterns during interview progression

Technical Implementation:
- Pandas-based data aggregation across participants per demographic group
- NumPy array operations for efficient trajectory computation
- Matplotlib visualization with custom styling and quadrant annotations
- Automatic file discovery and robust error handling for missing data

Data Processing Pipeline:
1. Load participant demographics and PHQ-8 binary classifications
2. Discover and load individual transcript files with VA scores
3. Group participants by gender and depression status
4. Aggregate VA trajectories within each demographic group
5. Generate 2D trajectory plots with temporal color coding
6. Save high-resolution plots for each demographic group

Output Files:
- Male_PHQ_Binary_0.png: Non-depressed male trajectory
- Male_PHQ_Binary_1.png: Depressed male trajectory
- Female_PHQ_Binary_0.png: Non-depressed female trajectory
- Female_PHQ_Binary_1.png: Depressed female trajectory
- Average_PHQ_Binary_0.png: Combined non-depressed trajectory
- Average_PHQ_Binary_1.png: Combined depressed trajectory

Research Applications:
- Visual validation of VA-based depression classification
- Identification of characteristic emotional trajectories for different populations
- Clinical interpretation support for VA-based assessment tools
- Publication-ready figures demonstrating temporal emotional dynamics

Clinical Significance:
- Reveals how emotional states evolve during clinical interviews
- Provides visual evidence of depression-related emotional patterns
- Supports development of trajectory-based diagnostic indicators
- Enables clinician training on VA pattern recognition

Related Files:
- va_classifier.visualizations.metrics — Quantitative temporal analysis
- va_classifier.visualizations.create_grid — Demographic comparison grids
- test_split.csv — Participant demographics and labels
- *_Transcript.csv — Individual VA-annotated interview transcripts

Usage:
    python visualization_2d.py
    # Requires: test_split.csv and *_Transcript.csv files in working directory
    # Outputs: High-resolution trajectory plots in plots/ directory

Note: These trajectory visualizations provide intuitive clinical interpretation
      of VA patterns, showing how emotional journeys differ between depressed
      and non-depressed populations during clinical interviews.
"""

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
    df = pd.read_csv(fp)
    df = df.reset_index().rename(columns={'index': 'Step'})
    transcripts[pid] = df

avg0_ids = labels[labels['PHQ_Binary'] == 0]['Participant_ID'].tolist()
avg1_ids = labels[labels['PHQ_Binary'] == 1]['Participant_ID'].tolist()

print("=== Checking group membership ===")
print("Average PHQ Binary 0: IDs =", sorted(avg0_ids))
print("Average PHQ Binary 1: IDs =", sorted(avg1_ids))

groups = {
    'Male PHQ Binary 0':    labels[(labels['Gender']=='male')   & (labels['PHQ_Binary']==0)]['Participant_ID'].tolist(),
    'Male PHQ Binary 1':    labels[(labels['Gender']=='male')   & (labels['PHQ_Binary']==1)]['Participant_ID'].tolist(),
    'Female PHQ Binary 0':  labels[(labels['Gender']=='female') & (labels['PHQ_Binary']==0)]['Participant_ID'].tolist(),
    'Female PHQ Binary 1':  labels[(labels['Gender']=='female') & (labels['PHQ_Binary']==1)]['Participant_ID'].tolist(),
    'Average PHQ Binary 1': avg1_ids,
    'Average PHQ Binary 0': avg0_ids,
}

output_folder = 'plots'
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
    norm = Normalize(vmin=0, vmax=max_steps - 1)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.plot(
        mean_valence.values,
        mean_arousal.values,
        color='gray',
        alpha=0.3,
        linewidth=2,
        zorder=1
    )
    scatter = ax.scatter(
        mean_valence.values,
        mean_arousal.values,
        c=steps,
        cmap='RdBu_r',
        norm=norm,
        s=20,
        edgecolor='k',
        zorder=2
    )

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)

    ax.set_xlabel('Mean Valence', fontsize=14)
    ax.set_ylabel('Mean Arousal', fontsize=14)
    ax.xaxis.set_label_coords(0.5, -0.05, transform=ax.transAxes)
    ax.yaxis.set_label_coords(-0.05, 0.5, transform=ax.transAxes)

    ax.text(0.25, 0.45, 'Happiness',
            ha='center', va='bottom', fontsize=12, color='darkgreen',
            transform=ax.transData)
    ax.text(-0.25, 0.45, 'Anger',
            ha='center', va='bottom', fontsize=12, color='darkred',
            transform=ax.transData)
    ax.text(-0.25, -0.45, 'Sadness',
            ha='center', va='top', fontsize=12, color='navy',
            transform=ax.transData)
    ax.text(0.25, -0.45, 'Tenderness',
            ha='center', va='top', fontsize=12, color='purple',
            transform=ax.transData)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Step (0 → red, end → blue)', fontsize=12)

    ax.set_title(f'{group_name}', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    filename = group_name.replace(' ', '_') + '.png'
    filepath = os.path.join(output_folder, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved '{group_name}' plot as '{filepath}'")
