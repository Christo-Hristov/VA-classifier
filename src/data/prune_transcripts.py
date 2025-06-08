# This script prunes transcripts based on length and/or valence-arousal scores
# Input: CSV file containing participant IDs and path to directory with transcripts w/ valence/arousal scores
# Output: Pruned transcripts are saved to length/va/length and va pruned directories in src/data/test_split directory
#   - length_pruned/: Transcripts pruned by word count
#   - va_pruned/: Transcripts pruned by valence-arousal scores
#   - length_and_VA/: Transcripts pruned by both criteria

# Call this function with: python prune_transcripts.py [FULL PATH TO THE SPLIT YOU WANT TO PRUNE] [--length] [--va] [--both]
# For example, to produce length pruned transcripts only for the test_split.csv file:
# python prune_transcripts.py /Users/kevinawang/Documents/GitHub/VA-classifier/src/data/test_split.csv --length
import os
import shutil
import pandas as pd
import argparse

# Function to create directory structure
def create_directory_structure(base_path, csv_name):
    # Create base directory
    os.makedirs(base_path, exist_ok=True)
    
    # Create subdirectories
    length_pruned_dir = os.path.join(base_path, 'length_pruned')
    va_pruned_dir = os.path.join(base_path, 'va_pruned')
    os.makedirs(length_pruned_dir, exist_ok=True)
    os.makedirs(va_pruned_dir, exist_ok=True)
    
    return length_pruned_dir, va_pruned_dir

# Function to copy input CSV
def copy_input_csv(input_csv, destination_dir):
    shutil.copy(input_csv, destination_dir)

# Function to prune transcripts by length
def prune_by_length(participant_id, transcripts_dir, output_dir):
    transcript_file = os.path.join(transcripts_dir, f'{participant_id}_Transcript.csv')
    if os.path.exists(transcript_file):
        df = pd.read_csv(transcript_file)
        # Count words in each row's Text column
        df['word_count'] = df['Text'].str.split().str.len()
        # Calculate mean word count for this transcript
        mean_word_count = df['word_count'].mean()
        # Keep rows with word count greater than the mean
        pruned_df = df[df['word_count'] > mean_word_count]
        # Drop the temporary word_count column
        pruned_df = pruned_df.drop('word_count', axis=1)
        pruned_df.to_csv(os.path.join(output_dir, f'{participant_id}_Transcript.csv'), index=False)

def prune_by_va(participant_id, transcripts_dir, output_dir):
    transcript_file = os.path.join(transcripts_dir, f'{participant_id}_Transcript.csv')
    if os.path.exists(transcript_file):
        df = pd.read_csv(transcript_file)
        # Filter by valence and arousal: Score ranges are the 25th-75th percentile for depressed patients from Yalcin's analysis
        pruned_df = df[(df['valence'].between(-0.110732, 0.040102)) & (df['arousal'].between(-0.057169, 0.058140))]
        pruned_df.to_csv(os.path.join(output_dir, f'{participant_id}_Transcript.csv'), index=False)

def process_transcripts(input_csv, transcripts_dir, length_pruned_dir, va_pruned_dir, length_and_va_dir, prune_length, prune_va, prune_both):
    # Read participant IDs from input CSV
    participant_ids = pd.read_csv(input_csv)['Participant_ID'].tolist()

    # Prune transcripts by length if flag is set
    if prune_length:
        for participant_id in participant_ids:
            prune_by_length(participant_id, transcripts_dir, length_pruned_dir)

    # Prune transcripts by valence-arousal if flag is set
    if prune_va:
        for participant_id in participant_ids:
            prune_by_va(participant_id, transcripts_dir, va_pruned_dir)

    # Prune transcripts by both length and valence-arousal if flag is set
    if prune_both:
        for participant_id in participant_ids:
            prune_by_both(participant_id, transcripts_dir, length_and_va_dir)

def main():
    parser = argparse.ArgumentParser(description='Prune transcripts by length or valence-arousal scores.')
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file containing participant IDs.')
    parser.add_argument('--length', action='store_true', help='Prune transcripts by text length.')
    parser.add_argument('--va', action='store_true', help='Prune transcripts by valence-arousal scores.')
    parser.add_argument('--both', action='store_true', help='Prune transcripts by both text length and valence-arousal scores.')
    args = parser.parse_args()

    # Define paths
    input_csv = args.input_csv
    csv_name = os.path.basename(input_csv).replace('.csv', '')
    base_path = os.path.join('/Users/kevinawang/Documents/GitHub/VA-classifier/src/data', csv_name)
    transcripts_dir = '/Users/kevinawang/Documents/GitHub/VA-classifier/data/edaic_transcripts/model1_outputted_va_scores'

    # Create directory structure
    length_pruned_dir, va_pruned_dir = create_directory_structure(base_path, csv_name)
    length_and_va_dir = os.path.join(base_path, 'length_and_VA')
    os.makedirs(length_and_va_dir, exist_ok=True)

    # Copy input CSV
    copy_input_csv(input_csv, base_path)

    # Process transcripts
    process_transcripts(input_csv, transcripts_dir, length_pruned_dir, va_pruned_dir, length_and_va_dir, args.length, args.va, args.both)

if __name__ == "__main__":
    main()
