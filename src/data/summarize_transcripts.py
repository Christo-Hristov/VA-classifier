"""
summarize_transcripts.py  ―  Transcript Summarizer
------------------------------------------------
This script processes clinical interview transcripts and generates concise summaries by:
1. Identifying and preserving the most important sentences
2. Summarizing less important sentences into a single entry
3. Preserving original VA scores for important sentences and averaging VA scores for summarized content

To run this script from the src directory use:
    python summarize_transcripts.py
"""

import os
import pandas as pd
from openai import OpenAI
import numpy as np
import json

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
    You are a clinical psychologist analyzing a transcript from a diagnostic interview.
    For each line in the transcript:
    1. Determine if it contains important clinical information that should be preserved verbatim
    2. If not important, mark it for summarization
    Return a JSON with two lists:
    - "important_lines": List of indices of important lines to preserve
    - "summary": A concise summary of the remaining lines
    """.strip()
    
    processed_count = 0
    for participant_id in participant_ids:
        transcript_file = os.path.join(transcripts_dir, f'{participant_id}_Transcript.csv')
        if os.path.exists(transcript_file):
            df = pd.read_csv(transcript_file)
            texts = df['Text'].tolist()
            valence_scores = df['valence'].tolist()
            arousal_scores = df['arousal'].tolist()

            # Use o4-mini model to identify important lines and summarize others
            response = client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(texts)},
                ]
            ).choices[0].message.content.strip()

            print(f"Response: {response}")

            # Strip code fencing if present — added for debugging
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "").strip()
            elif response.startswith("```"):
                response = response.replace("```", "").strip()

            # Parse response to get important lines and summary
            result = json.loads(response)
            important_indices = result["important_lines"]
            summary = result["summary"]

            # Create new DataFrame with important lines and summary
            important_texts = [texts[i] for i in important_indices if i < len(texts)]
            important_valence = [valence_scores[i] for i in important_indices]
            important_arousal = [arousal_scores[i] for i in important_indices]

            # Calculate average VA scores for summarized content
            summary_indices = [i for i in range(len(texts)) if i not in important_indices]
            avg_valence = np.mean([valence_scores[i] for i in summary_indices]) if summary_indices else 0
            avg_arousal = np.mean([arousal_scores[i] for i in summary_indices]) if summary_indices else 0

            # Combine important lines and summary
            final_texts = important_texts + [summary]
            final_valence = important_valence + [avg_valence]
            final_arousal = important_arousal + [avg_arousal]

            # Create DataFrame and save
            summary_df = pd.DataFrame({
                'Text': final_texts,
                'valence': final_valence,
                'arousal': final_arousal
            })

            output_path = os.path.join(output_dir, f'{participant_id}_Transcript.csv')
            summary_df.to_csv(output_path, index=False)
            processed_count += 1
            print(f"Transcript for patient {participant_id} has been processed")

    print(f"\nAll transcripts have been summarized and output to {output_dir}")
    print(f"Total transcripts processed: {processed_count}")

if __name__ == "__main__":
    summarize_test_transcripts() 