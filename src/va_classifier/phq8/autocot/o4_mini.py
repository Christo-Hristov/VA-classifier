"""
PHQ-8 AutoCoT o4-mini Implementation with VA Enhancement (§3.2.2)
================================================================

Purpose: Configurable AutoCoT implementation using GPT-4o-mini for PHQ-8 depression prediction
         with optional valence-arousal context and multiple experimental configurations.
         This is the most flexible AutoCoT implementation supporting various pruning strategies.

Paper Section: §3.2.2 AutoCoT Few-Shot Prompting
Model: GPT-4o-mini (cost-effective alternative to GPT-4)

This script implements a comprehensive, configurable AutoCoT pipeline:
1. Demo Creation: Select N participants (configurable) with transcripts and VA scores
2. Rationale Generation: Generate reasoning demonstrations with/without VA context
3. Demo Formatting: Format demonstrations for AutoCoT prompting
4. Test Processing: Remove demo participants, prepare evaluation set
5. Configurable Inference: Run AutoCoT with various experimental conditions
6. Statistical Evaluation: Calculate MAE/RMSE metrics

Key Configuration Options:
- --without_va: Run without valence-arousal context (baseline comparison)
- --summary: Use transcript summaries instead of full text
- --length_pruned: Use length-based pruned transcripts
- --va_pruned: Use VA-based pruned transcripts (remove low-emotion segments)
- --n: Number of demonstration examples (2, 4, 6, 8)

Experimental Conditions Supported:
- Baseline AutoCoT (no VA context)
- VA-enhanced AutoCoT (with valence-arousal scores)
- Summarization experiments (condensed transcripts)
- Length pruning (remove excessive content)
- VA pruning (emotion-based content selection)
- Demo count optimization (n=2,4,6,8 demonstrations)

AutoCoT Strategy:
- Uses GPT-4o-mini for cost-effective experimentation
- Line-by-line transcript processing with optional VA annotations
- Flexible demonstration generation based on experimental condition
- Systematic output organization by experimental configuration

Input: E-DAIC clinical interview transcripts with optional VA scores
Output: PHQ-8 total scores (0-24 scale) with comprehensive experimental logging

Experimental Design:
- Supports systematic comparison across multiple conditions
- Organized output structure: outputs/{va|no_VA}/{condition}/
- Enables analysis of VA impact, pruning effectiveness, demo optimization
- Cost-effective experimentation using o4-mini vs GPT-4

Clinical Applications:
- Tests scalability of AutoCoT approach with smaller models
- Evaluates impact of transcript preprocessing strategies
- Optimizes demonstration count for clinical deployment
- Assesses VA context value for depression prediction

Related Files:
- va_classifier.phq8.autocot.baseline — Simplified baseline implementation
- va_classifier.phq8.autocot.with_va — Full GPT-4 VA implementation
- va_classifier.phq8.autocot.detailed — Comprehensive question-level scoring
- va_classifier.data_prep.transcript_pruning — Preprocessing utilities

Usage Examples:
    # Baseline without VA
    python o4_mini.py --without_va --n 8
    
    # VA-enhanced with summarization
    python o4_mini.py --summary --n 6
    
    # Length-pruned with VA context
    python o4_mini.py --length_pruned --n 4
    
    # VA-pruned experimental condition
    python o4_mini.py --va_pruned --n 2

Results Structure:
    outputs/{va|no_VA}/{condition}/
    ├── demo_set.csv                 # Selected demonstration examples
    ├── rationale_demos.csv          # Generated reasoning chains
    ├── formatted_demos.csv          # AutoCoT-formatted demonstrations
    ├── new_test_split.csv          # Evaluation participants
    ├── autocotVA_predictions.csv   # Model predictions
    └── statistics_results.txt      # Performance metrics (MAE, RMSE)
"""

import os
import pandas as pd
from openai import OpenAI
import re
import json
from tqdm import tqdm
import numpy as np
import argparse

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def create_demo_set(TRANSCRIPT_DIR, args):
    """
    Creates a demo set to be ran through zero shot for prompt. Done by 
    selecting 8 participants from test split and merging their transcripts, 
    VA scores, and PHQ scores.
    
    Returns:
        pd.DataFrame: DataFrame containing demo examples with merged data
    """
    print(f"Transcript directory used: {TRANSCRIPT_DIR}")
    # Load test split to get participant IDs
    test_split_path = "/Users/kevinawang/Documents/GitHub/VA-classifier/src/data/test_split.csv"
    test_df = pd.read_csv(test_split_path)
    print(f"Loaded test split with {len(test_df)} participants")
    
    # Use the --n flag to determine the number of participants
    selected_ids = test_df['Participant_ID'].sample(n=args.n, random_state=42).tolist()
    print(f"Selected demo IDs: {selected_ids}")
    demo_data = []
    
    for participant_id in selected_ids:
        try:
            # Load transcript
            transcript_path = os.path.join(TRANSCRIPT_DIR, f'{participant_id}_Transcript.csv')
            transcript_df = pd.read_csv(transcript_path)
            
            # Load VA scores
            va_path = os.path.join(TRANSCRIPT_DIR, f'{participant_id}_Transcript.csv')
            va_df = pd.read_csv(va_path)
            
            # Get PHQ score for this participant
            phq_score = test_df[test_df['Participant_ID'] == participant_id]['PHQ_Score'].iloc[0]
            
            # Create demo example
            demo_example = {
                'Participant_ID': participant_id,
                'Transcript': transcript_df['Text'].tolist(),  # Array of transcript lines
                'VA_Scores': va_df[['valence', 'arousal']].values.tolist(),  # Array of [valence, arousal] pairs
                'PHQ_Score': phq_score
            }
            
            demo_data.append(demo_example)
            print(f"✅ Successfully added participant {participant_id} to demo set")
            
        except Exception as e:
            print(f"⚠️ Error processing participant {participant_id}: {str(e)}")
            # If we encounter an error, we'll need to select a replacement ID
            remaining_ids = set(test_df['Participant_ID']) - set(selected_ids)
            if not remaining_ids:
                raise Exception("No more participant IDs available for selection")
            
            # Select a new ID from remaining IDs
            new_id = pd.Series(list(remaining_ids)).sample(n=1, random_state=42).iloc[0]
            selected_ids.append(new_id)
            print(f"🔄 Selected replacement participant ID: {new_id}")
    
    # Convert to DataFrame
    demo_df = pd.DataFrame(demo_data)
    
    # Save demo set
    output_path = '/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT-o4-mini/outputs/demo_set.csv'
    demo_df.to_csv(output_path, index=False)
    print(f"✅ Demo set saved to {output_path}")
    
    return demo_df

def generate_rationale_demos_with_gpt(demo_df, output_json_path, num_demos=8, model="o4-mini", args=None):
    """
    Generates rationale demos using zero-shot CoT and saves them to a JSON file.
    
    Args:
        demo_df (pd.DataFrame): DataFrame containing demo examples with merged data
        output_json_path (str): Path to save the generated rationale demos as a JSON file.
        num_demos (int): Number of demos to generate.
        model (str): Model name for o4-mini.
    
    Outputs:
        Saves rationale demos to a JSON file.
    """
    if args.without_va:
        SYSTEM_PROMPT = """
            You are a clinical psychologist. 
            You read transcripts of a patient from a diagnostic interview and estimate the PHQ-8 score (0-24) by reasoning through what the participant said. 
            You will be given what the patient said line by line.
            Return only your reasoning — do not include the final score.
            """.strip()
    else:
        SYSTEM_PROMPT = """
            You are a clinical psychologist. 
            You read transcripts of a patient from a diagnostic interview and estimate the PHQ-8 score (0-24) by reasoning through what the participant said. 
            You will be given what the patient said line by line, along with the valence and arousal score (-1 to 1) for that line. 
            Return only your reasoning — do not include the final score.
            """.strip()
    
    # Sample from the DataFrame directly instead of reading from CSV
    df = demo_df.sample(frac=1, random_state=42).head(num_demos)
    demo_list = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Format transcript with VA scores
        transcript_lines = []
        for text, va_scores in zip(row['Transcript'], row['VA_Scores']):
            if args.without_va:
                transcript_lines.append(f"Line: {text}")
            else:
                transcript_lines.append(f"Line: {text}\nValence: {va_scores[0]}, Arousal: {va_scores[1]}")
        
        if args.without_va:
            user_prompt = (
                f"Transcript:\n" + "\n".join(transcript_lines) + "\n\n"
                f"PHQ-8 Score: {row['PHQ_Score']}\n"
                "Q: Explain how the transcript content supports this PHQ-8 score.\n"
                "A: Let's think step by step."
            )
        else:
            user_prompt = (
                f"Transcript with Valence-Arousal Scores:\n" + "\n".join(transcript_lines) + "\n\n"
                f"PHQ-8 Score: {row['PHQ_Score']}\n"
                "Q: Explain how the transcript content and emotional patterns (valence/arousal) support this PHQ-8 score.\n"
                "A: Let's think step by step."
            )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
            )

            rationale = response.choices[0].message.content.strip()

            demo_list.append({
                "question": user_prompt,
                "rationale": rationale,
                "phq_score": row['PHQ_Score']
            })

        except Exception as e:
            print(f"[ERROR] GPT failed on PID {row['Participant_ID']} → {e}")
            continue

    # Save output
    with open(output_json_path, "w") as f:
        json.dump({"demo": demo_list}, f, indent=4)
    print(f"✅ Saved {len(demo_list)} demo rationales to: {output_json_path}")


def format_autocot_demo_text(json_path, output_csv_path=None, args=None):
    """
    Formats demo examples from GPT rationale output into Auto-CoT compatible Q&A format.
    
    Args:
        json_path (str): Path to the JSON file containing demo rationales.
        output_csv_path (str, optional): Path to save the formatted demos as a CSV file.
    
    Returns:
        list: A list of formatted demo strings.
    """
    # Read the rationale demos JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    formatted_demos = []
    
    system_prompt = "You are a clinical psychologist. You read transcripts of a patient from a diagnostic interview and estimate the PHQ-8 score (0-24) by reasoning through what the participant said. "
    if args.without_va:
        system_prompt += "You will be given what the patient said line by line."
    else:
        system_prompt += "You will be given what the patient said line by line, along with the valence and arousal score (-1 to 1) for that line."
    
    for demo in data['demo']:
        question = system_prompt
        rationale = demo['rationale']
        phq_score = demo['phq_score']
        
        formatted = (
            f"Q: {question}\n"
            f"{rationale}. Therefore, the answer is {phq_score}"
        )
        formatted_demos.append(formatted)

    if output_csv_path:
        pd.DataFrame({"Formatted Demo": formatted_demos}).to_csv(output_csv_path, index=False)
        print(f"Formatted demos saved to {output_csv_path}")

    return formatted_demos

def create_new_test_split(demo_csv_path, original_test_split_path, output_path, TRANSCRIPT_DIR):
    """
    Creates a new test split by removing the participant IDs that were used for demos,
    and coalesces the remaining transcripts with VA scores and PHQ-8 total score.
    
    Args:
        demo_csv_path (str): Path to the CSV file containing demo participants
        original_test_split_path (str): Path to the original test split CSV file
        output_path (str): Path to save the new test split CSV file
    """
    # Read the demo CSV file
    demo_df = pd.read_csv(demo_csv_path)
    
    # Get the list of participant IDs used in demos
    demo_pids = set(demo_df['Participant_ID'].unique())
    
    # Read the original test split
    test_split_df = pd.read_csv(original_test_split_path)
    
    # Remove demo participants from test split
    new_test_split = test_split_df[~test_split_df['Participant_ID'].isin(demo_pids)]
    
    # Process remaining transcripts
    processed_transcripts = []
    
    for _, row in tqdm(new_test_split.iterrows(), desc="Processing remaining test transcripts"):
        participant_id = row['Participant_ID']
        
        try:
            # Load transcript
            transcript_path = os.path.join(TRANSCRIPT_DIR, f'{participant_id}_Transcript.csv')
            transcript_df = pd.read_csv(transcript_path)
            
            # Load VA scores
            va_path = os.path.join(TRANSCRIPT_DIR, f'{participant_id}_Transcript.csv')
            va_df = pd.read_csv(va_path)
            
            # Get PHQ score for this participant
            phq_score = row['PHQ_Score']
            
            # Create processed example
            processed_example = {
                'Participant_ID': participant_id,
                'Transcript': transcript_df['Text'].tolist(),  # Array of transcript lines
                'VA_Scores': va_df[['valence', 'arousal']].values.tolist(),  # Array of [valence, arousal] pairs
                'PHQ_Score': phq_score
            }
            
            processed_transcripts.append(processed_example)
            print(f"✅ Successfully processed participant {participant_id}")
            
        except Exception as e:
            print(f"⚠️ Skipping participant {participant_id} due to error: {str(e)}")
            continue
    
    # Convert to DataFrame and save
    processed_df = pd.DataFrame(processed_transcripts)
    processed_df.to_csv(output_path, index=False)
    
    print(f"✅ New test split saved to: {output_path}")
    print(f"Removed {len(demo_pids)} demo participants from test split")
    print(f"New test split contains {len(processed_df)} participants")
    
    return processed_df


def run_autocot_inference(output_path, model="o4-mini", args=None):
    """
    Runs Auto-CoT inference on test data and saves predictions incrementally.
    
    Args:
        output_path (str): Path to save the predictions.
        model (str): Model name for o4-mini.
    
    Outputs:
        Saves predictions to a CSV file with columns: Participant_ID, Response, and PHQ-8 Total score.
        Updates the file after each participant is processed.
    """
    # Load the formatted demos and test data
    formatted_demos = pd.read_csv("/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT_with_VA/outputs/formatted_demos.csv")
    test_df = pd.read_csv("/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT_with_VA/outputs/new_test_split.csv")
    
    # Create or load existing predictions file
    if os.path.exists(output_path):
        predictions_df = pd.read_csv(output_path)
        processed_ids = set(predictions_df['Participant_ID'])
    else:
        predictions_df = pd.DataFrame(columns=['Participant_ID', 'Response', 'PHQ_8Total'])
        processed_ids = set()

    # Iterate through each test transcript
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running Auto-CoT Inference"):
        participant_id = row["Participant_ID"]
            
        transcript = row["Transcript"]
        va_scores = row["VA_Scores"]

        # Print progress info for tracking
        print(f"\n🔍 Processing Participant_ID: {participant_id} ({idx+1}/{len(test_df)})")

        # Format transcript with VA scores
        formatted_transcript = ""
        for text, va in zip(eval(transcript), eval(va_scores)):
            formatted_transcript += f"Line: {text}\nValence: {va[0]:.2f}, Arousal: {va[1]:.2f}\n\n"

        # Construct the full prompt using demo rationales + the current transcript
        if args.without_va:
            prompt = (
                "\n\n".join(formatted_demos['Formatted Demo'].tolist()) + "\n\n"
                "Q: You are a clinical psychologist. You read transcripts of a patient from a diagnostic interview and estimate the PHQ-8 score (0-24) by reasoning through what the participant said. You will be given what the patient said line by line. "
                "Evaluate the patient's total PHQ8 score\n"
                f"{formatted_transcript}\n"
                "A: Let's think step by step. After analyzing the transcript, you MUST output the score in this exact format:\n"
                "PHQ_8Total: [score]\n"
                "Again, make sure to output the score in the above format."
            )
        else:
            prompt = (
                "\n\n".join(formatted_demos['Formatted Demo'].tolist()) + "\n\n"
                "Q: You are a clinical psychologist. You read transcripts of a patient from a diagnostic interview and estimate the PHQ-8 score (0-24) by reasoning through what the participant said. You will be given what the patient said line by line, along with the valence and arousal score (-1 to 1) for that line. "
                "Evaluate the patient's total PHQ8 score\n"
                f"{formatted_transcript}\n"
                "A: Let's think step by step. After analyzing the transcript, you MUST output the score in this exact format:\n"
                "PHQ_8Total: [score]\n"
                "Again, make sure to output the score in the above format."
            )

        try:
            # Query o4-mini for reasoning and prediction
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            reply = response.choices[0].message.content.strip()

            # Extract total score from the response using regex
            match = re.search(r"PHQ_8Total:\s*(\d+)", reply)
            total_score = float(match.group(1)) if match else None

            # Create new prediction row
            new_prediction = pd.DataFrame([{
                "Participant_ID": participant_id,
                "Response": reply,
                "PHQ_8Total": total_score
            }])

            # Append to existing predictions and save immediately
            predictions_df = pd.concat([predictions_df, new_prediction], ignore_index=True)
            predictions_df.to_csv(output_path, index=False)
            print(f"✅ Saved prediction for Participant_ID: {participant_id}")

        except Exception as e:
            # Handle o4-mini failure gracefully
            print(f"[ERROR] o4-mini failed on PID {participant_id} → {e}")
            new_prediction = pd.DataFrame([{
                "Participant_ID": participant_id,
                "Response": "",
                "PHQ_8Total": None
            }])
            predictions_df = pd.concat([predictions_df, new_prediction], ignore_index=True)
            predictions_df.to_csv(output_path, index=False)

    print(f"\n✅ Finished! All predictions saved to: {output_path}")


def calculate_statistics(predictions_csv_path, test_split_path):
    """
    Calculates MAE and RMSE for PHQ-8 total score.
    
    Args:
        predictions_csv_path (str): Path to the predictions CSV file.
        test_split_path (str): Path to the test split CSV file containing ground truth PHQ scores.
    
    Returns:
        tuple: MAE and RMSE for total PHQ-8 score.
    """
    # Load the predictions and test split CSVs
    predictions_df = pd.read_csv(predictions_csv_path)
    test_df = pd.read_csv(test_split_path)
    
    # Merge on Participant_ID to align gold and predicted scores
    merged_df = pd.merge(
        test_df[['Participant_ID', 'PHQ_Score']], 
        predictions_df[['Participant_ID', 'PHQ_8Total']], 
        on='Participant_ID', 
        suffixes=('_gold', '_pred')
    )
    
    gold_scores = merged_df['PHQ_Score']
    predicted_scores = merged_df['PHQ_8Total']
    
    # Drop NaN values
    valid_indices = ~predicted_scores.isna()
    gold_scores = gold_scores[valid_indices]
    predicted_scores = predicted_scores[valid_indices]
    
    # Calculate MAE and RMSE
    mae = np.mean(np.abs(gold_scores - predicted_scores))
    rmse = np.sqrt(np.mean((gold_scores - predicted_scores) ** 2))
    
    # Print statistics
    print(f"PHQ-8 Total Score - MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    
    # Save results to a file
    results_output_path = '/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT_with_VA/outputs/statistics_results.txt'
    with open(results_output_path, 'w') as f:
        f.write("PHQ-8 Total Score Statistics:\n")
        f.write(f"MAE: {mae:.3f}\n")
        f.write(f"RMSE: {rmse:.3f}\n")
    print(f"Results saved to {results_output_path}")
    
    return mae, rmse

def main():
    parser = argparse.ArgumentParser(description='Auto-CoT with VA inference')
    parser.add_argument('--transcripts', default=None,
                        help='Folder that contains the <PID>_Transcript.csv files')
    parser.add_argument('--summary', action='store_true',
                        help='Use existing VA scores from input CSV instead of computing new ones')
    parser.add_argument('--length_pruned', action='store_true',
                        help='Use length pruned transcripts with existing VA scores from input CSV instead of computing new ones')
    parser.add_argument('--va_pruned', action='store_true',
                        help='Use VA pruned transcripts with existing VA scores from input CSV instead of computing new ones')
    parser.add_argument('--without_va', action='store_true',
                        help='Only use transcript text without VA scores')
    parser.add_argument('--n', type=int, default=8,
                        help='Number of participants to select for demo creation')

    args = parser.parse_args()

    TRANSCRIPT_DIR = args.transcripts or '/Users/kevinawang/Documents/GitHub/VA-classifier/data/edaic_transcripts/model1_outputted_va_scores'

    if args.length_pruned:
        TRANSCRIPT_DIR = '/Users/kevinawang/Documents/GitHub/VA-classifier/data/edaic_transcripts/pruned_transcripts/length_pruned'
    elif args.va_pruned:
        TRANSCRIPT_DIR = '/Users/kevinawang/Documents/GitHub/VA-classifier/data/edaic_transcripts/pruned_transcripts/va_pruned'
    elif args.summary:
        TRANSCRIPT_DIR = '/Users/kevinawang/Documents/GitHub/VA-classifier/data/edaic_transcripts/summarized_transcripts'

    # Determine base output directory based on VA usage
    va_usage_dir = "va" if not args.without_va else "no_VA"
    base_path = "/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT-o4-mini"
    base_output_dir = os.path.join(base_path, "outputs", va_usage_dir)

    # Determine subdirectory based on specific arguments
    if args.summary:
        sub_dir = "summary"
    elif args.length_pruned:
        sub_dir = "prune_length"
    elif args.va_pruned:
        sub_dir = "prune_va"
    else:
        sub_dir = "default"

    # Combine base and subdirectory to form the final output directory
    final_output_dir = os.path.join(base_output_dir, sub_dir)
    os.makedirs(final_output_dir, exist_ok=True)

    # Update paths to use the final output directory
    rationale_demos_path = os.path.join(final_output_dir, "rationale_demos.csv")
    formatted_demos_path = os.path.join(final_output_dir, "formatted_demos.csv")
    demo_set_path = os.path.join(final_output_dir, "demo_set.csv")
    new_test_path = os.path.join(final_output_dir, "new_test_split.csv")
    predictions_path = os.path.join(final_output_dir, "autocotVA_predictions.csv")
    results_output_path = os.path.join(final_output_dir, "statistics_results.txt")

    # Ensure all necessary directories are created before file operations
    os.makedirs(os.path.dirname(demo_set_path), exist_ok=True)

    # Use the updated paths in the script
    print("Creating demo set...🫠")
    demo_df = create_demo_set(TRANSCRIPT_DIR, args)
    
    # Save demo set to the correct directory
    demo_df.to_csv(demo_set_path, index=False)
    print(f"✅ Demo set saved to {demo_set_path}")
    
    # Generate and format demos
    print("Calling generate_rationale_demos_with_gpt...🫠")
    generate_rationale_demos_with_gpt(demo_df, rationale_demos_path, args=args)
    
    print("Calling format_autocot_demo_text...🫠")
    formatted_demos = format_autocot_demo_text(rationale_demos_path, formatted_demos_path, args=args)
    
    # Create test split
    original_test_path = "/Users/kevinawang/Documents/GitHub/VA-classifier/src/data/test_split.csv"
    create_new_test_split(demo_set_path, original_test_path, new_test_path, TRANSCRIPT_DIR)
    
    # Run inference and calculate statistics
    print("Calling run_autocot_inference...🫠")
    run_autocot_inference(predictions_path, args=args)
    
    print("Calling calculate_statistics...🫠")
    mae, rmse = calculate_statistics(predictions_path, new_test_path)

if __name__ == "__main__":
    main()