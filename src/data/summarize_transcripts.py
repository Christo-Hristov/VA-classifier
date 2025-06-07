"""
summarize_transcripts.py  ―  Transcript Summarizer
------------------------------------------------
This script processes clinical interview transcripts and generates concise summaries using OpenAI's o4-mini model.
For each transcript:
1. Loads the transcript and its associated valence/arousal scores
2. Uses o4-mini to generate a ~250 word summary from a clinical psychologist's perspective
3. Saves the summary along with the original VA scores (line by line) to a new CSV file

To run this script from the src directory use:
    python summarize_transcripts.py
"""

import os
import pandas as pd
from openai import OpenAI

def summarize_test_transcripts():
    # OpenAI call set up
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # Data processing set up
    test_split_path = '/Users/kevinawang/Documents/GitHub/VA-classifier/src/data/test_split.csv'
    transcripts_dir = '/Users/kevinawang/Documents/GitHub/VA-classifier/data/edaic_transcripts/model1_outputted_va_scores'
    output_dir = '/Users/kevinawang/Documents/GitHub/VA-classifier/data/edaic_transcripts/summarized_transcripts'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load participant IDs from the test set CSV
    participant_ids = pd.read_csv(test_split_path)['Participant_ID'].tolist()

    SYSTEM_PROMPT = """
    You are a clinical psychologist.
    Summarize the main topics discussed by the patient in the transcript below. The transcript is from 
    a diagnostic psychological clinical interview.
    Make sure to keep details. Capture topics, the patient's tone, how the conversation progresses. 
    Your returned summary should be around 250 words. The transcript is as follows: 
    """.strip()
    
    processed_count = 0
    for participant_id in participant_ids:
        transcript_file = os.path.join(transcripts_dir, f'{participant_id}_Transcript.csv')
        if os.path.exists(transcript_file):
            df = pd.read_csv(transcript_file)
            text = ' '.join(df['Text'].tolist())
            valence_scores = df['valence'].tolist()
            arousal_scores = df['arousal'].tolist()

            # Use o4-mini model to generate a summary of the transcript
            response = client.chat.completions.create(
                model="o4-mini",
                messages =[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ]
            ).choices[0].message.content.strip()

            # Create a DataFrame for the summary
            summary_df = pd.DataFrame({
                'text': [response],
                'valence_score': [valence_scores],
                'arousal_score': [arousal_scores]
            })

            # Save the summary to the output directory
            output_path = os.path.join(output_dir, f'{participant_id}_summarized_transcript.csv')
            summary_df.to_csv(output_path, index=False)
            processed_count += 1
            print(f"Transcript for patient {participant_id} has been processed")

    print(f"\nAll transcripts have been summarized and output to {output_dir}")
    print(f"Total transcripts processed: {processed_count}")


if __name__ == "__main__":
    summarize_test_transcripts() 