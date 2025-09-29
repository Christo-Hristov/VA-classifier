"""
Temporal VA Trajectory Analysis by Demographics (§5.2 & §6 Discussion)

Purpose: Analyzes valence-arousal patterns across different temporal windows
         within clinical interviews, stratified by gender and depression status.
         Reveals temporal dynamics of emotional expression in clinical populations.

Paper Section: §5.2 PHQ-8 Results & §6 Discussion (temporal and demographic analysis)
Application: Understanding how emotional patterns evolve during clinical interviews

This script performs comprehensive temporal VA analysis across demographic groups:

1. Demographic Stratification:
   - Male PHQ 0: Non-depressed male participants
   - Female PHQ 0: Non-depressed female participants  
   - Male PHQ 1: Depressed male participants
   - Female PHQ 1: Depressed female participants

2. Temporal Window Analysis:
   - First 5 utterances: Initial emotional state and interview opening
   - Middle 5 utterances: Mid-interview emotional patterns
   - Last 5 utterances: Closing emotional state and interview conclusion

3. VA Metrics Calculated:
   - Average valence per temporal window per demographic group
   - Average arousal per temporal window per demographic group
   - Group-level aggregation with robust handling of missing data
   - Temporal progression analysis across interview segments

4. Clinical Insights Generated:
   - Emotional trajectory differences between depressed and non-depressed groups
   - Gender-specific patterns in emotional expression during interviews
   - Temporal dynamics revealing interview effects on emotional state
   - VA pattern stability vs variability across interview progression

Research Questions Addressed:
- Do depressed individuals show different emotional trajectories during interviews?
- Are there gender differences in how emotions evolve throughout clinical interviews?
- Does emotional state change systematically from interview beginning to end?
- Can temporal VA patterns provide additional diagnostic information?

Methodological Approach:
- Robust averaging with NaN handling for missing or incomplete transcripts
- Minimum length filtering (≥5 utterances) for reliable temporal analysis
- Systematic demographic grouping based on PHQ-8 binary classification
- Temporal window standardization across all participants

Expected Clinical Patterns:
- Depressed groups: Lower valence across all temporal windows
- Gender differences: Potential variations in arousal patterns
- Temporal effects: Possible emotional changes from interview start to end
- Diagnostic utility: Temporal patterns may enhance depression prediction

Technical Implementation:
- Pandas-based data processing with efficient grouping operations
- NumPy statistical calculations for robust mean computation
- Glob-based file discovery for systematic transcript processing
- Structured output format for further statistical analysis

Output Format:
    Group         Window    Avg Valence    Avg Arousal
    Male PHQ 0    First 5   -0.123         0.456
    Female PHQ 0  First 5   -0.089         0.234
    [Additional rows for all groups and windows]

Research Applications:
- Validates temporal consistency of VA-based depression markers
- Identifies optimal temporal windows for clinical assessment
- Supports demographic-specific clinical interpretation guidelines
- Enables development of temporally-aware diagnostic algorithms

Clinical Significance:
- Reveals how depression manifests temporally during clinical interviews
- Identifies gender-specific emotional expression patterns
- Supports development of interview protocols optimized for VA assessment
- Provides foundation for real-time emotional monitoring applications

Related Files:
- va_classifier.data.visualize_time_step — Temporal trajectory visualization
- va_classifier.visualizations.create_grid — Demographic comparison plots
- test_split.csv — Participant demographics and PHQ-8 labels
- *_Transcript.csv — Individual participant interview transcripts with VA scores

Usage:
    python metrics.py
    # Requires: test_split.csv and *_Transcript.csv files in working directory
    # Outputs: Comprehensive temporal VA analysis table

Note: This analysis provides critical temporal and demographic insights that
      complement the paper's main VA prediction results, revealing how emotional
      patterns evolve during clinical interviews across different populations.
"""

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
