import pandas as pd
import glob
import os
import numpy as np

labels = pd.read_csv('test_split.csv')

transcript_files = glob.glob('*_Transcript.csv')

transcripts = {}
for fp in transcript_files:
    pid = int(os.path.basename(fp).split('_')[0])
    df = pd.read_csv(fp).reset_index().rename(columns={'index': 'Step'})
    transcripts[pid] = df

groups = {
    'Male PHQ 0':   labels[(labels['Gender']=='male')   & (labels['PHQ_Binary']==0)]['Participant_ID'].tolist(),
    'Female PHQ 0': labels[(labels['Gender']=='female') & (labels['PHQ_Binary']==0)]['Participant_ID'].tolist(),
    'Male PHQ 1':   labels[(labels['Gender']=='male')   & (labels['PHQ_Binary']==1)]['Participant_ID'].tolist(),
    'Female PHQ 1': labels[(labels['Gender']=='female') & (labels['PHQ_Binary']==1)]['Participant_ID'].tolist(),
}

results = []

for group_name, id_list in groups.items():
    first5_val, mid5_val, last5_val = [], [], []
    first5_ar,  mid5_ar,  last5_ar  = [], [], []
    
    for pid in id_list:
        if pid not in transcripts:
            continue
        df = transcripts[pid]
        vals = df['valence'].values
        ars  = df['arousal'].values
        L = len(df)
        if L < 5:
            continue
        
        v_f = np.mean(vals[0:5])
        a_f = np.mean(ars[0:5])
        
        start_mid = (L - 5) // 2
        v_m = np.mean(vals[start_mid:start_mid + 5])
        a_m = np.mean(ars[start_mid:start_mid + 5])
        
        v_l = np.mean(vals[-5:])
        a_l = np.mean(ars[-5:])
        
        first5_val.append(v_f);  first5_ar.append(a_f)
        mid5_val.append(v_m);    mid5_ar.append(a_m)
        last5_val.append(v_l);   last5_ar.append(a_l)
    
    def grp_mean(lst):
        return np.nan if not lst else np.mean(lst)
    
    results.append({
        'Group': group_name,
        'Window': 'First 5',
        'Avg Valence': grp_mean(first5_val),
        'Avg Arousal': grp_mean(first5_ar)
    })
    results.append({
        'Group': group_name,
        'Window': 'Mid 5',
        'Avg Valence': grp_mean(mid5_val),
        'Avg Arousal': grp_mean(mid5_ar)
    })
    results.append({
        'Group': group_name,
        'Window': 'Last 5',
        'Avg Valence': grp_mean(last5_val),
        'Avg Arousal': grp_mean(last5_ar)
    })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
