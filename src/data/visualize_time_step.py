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
