import os
import pandas as pd
from openai import OpenAI
import re
import json
from tqdm import tqdm
import numpy as np

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def merge_and_split_transcripts_and_labels(transcripts_dir, labels_df, num_demo=8):
    """
    Merges transcripts and labels, then splits the data into demo and training datasets.
    
    Args:
        transcripts_dir (str): Directory containing transcript CSV files.
        labels_df (pd.DataFrame): DataFrame containing PHQ-8 labels.
        num_demo (int): Number of demo examples to select.
    
    Outputs:
        Saves demo and training datasets to CSV files.
    """
    transcripts = []
    
    # Iterate over each file in the transcripts directory
    for file_name in os.listdir(transcripts_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(transcripts_dir, file_name)
            df = pd.read_csv(file_path)
            
            # Extract Participant_ID from the filename
            participant_id = file_name.split('_')[0]
            
            # Concatenate all text from the transcript
            full_text = ' '.join(df['Text'].tolist())
            
            transcripts.append({'Participant_ID': participant_id, 'Text': full_text})
    
    # Create a DataFrame from transcripts
    transcripts_df = pd.DataFrame(transcripts)
    
    # Convert Participant_ID to string in both DataFrames
    transcripts_df['Participant_ID'] = transcripts_df['Participant_ID'].astype(str)
    labels_df['Participant_ID'] = labels_df['Participant_ID'].astype(str)
    
    # Merge with labels
    merged_df = pd.merge(transcripts_df, labels_df, on='Participant_ID', how='inner')
    
    # Select and reorder columns
    merged_df = merged_df[['Participant_ID', 'Text', 'PHQ_8NoInterest', 'PHQ_8Depressed',
                           'PHQ_8Sleep', 'PHQ_8Tired', 'PHQ_8Appetite', 'PHQ_8Failure',
                           'PHQ_8Concentrating', 'PHQ_8Moving', 'PHQ_8Total']]
    
    # Shuffle the DataFrame to ensure random selection
    shuffled_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Select the first `num_demo` rows for demo
    demo_df = shuffled_df.head(num_demo)
    
    # The rest are for training
    training_df = shuffled_df.iloc[num_demo:]
    
    # Output to CSV files
    output_dir = '/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT/outputs'
    demo_output_path = os.path.join(output_dir, 'demo_transcripts.csv')
    training_output_path = os.path.join(output_dir, 'training_transcripts.csv')
    demo_df.to_csv(demo_output_path, index=False)
    training_df.to_csv(training_output_path, index=False)
    
    print(f"Demo data saved to {demo_output_path}")
    print(f"Training data saved to {training_output_path}")


def generate_rationale_demos_with_gpt(demo_csv_path, output_json_path, num_demos=8, model="gpt-4o"):
    """
    Generates rationale demos using zero-shot CoT and saves them to a JSON file.
    
    Args:
        demo_csv_path (str): Path to the demo transcripts CSV file.
        output_json_path (str): Path to save the generated rationale demos as a JSON file.
        num_demos (int): Number of demos to generate.
        model (str): Model name for GPT-4o.
    
    Outputs:
        Saves rationale demos to a JSON file.
    """
    SYSTEM_PROMPT = """
        You are a clinical psychologist. 
        You read transcripts of a patient from a diagnostic interview and estimate the PHQ-8 score by reasoning through what the participant said. 
        Return only your reasoning — do not include the final score.
        """.strip()
    df = pd.read_csv(demo_csv_path).sample(frac=1, random_state=42).head(num_demos)
    demo_list = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_prompt = (
            f"Transcript: {row['Text']}\n"
            f"True PHQ-8 Scores:\n"
            f"- No Interest: {row['PHQ_8NoInterest']}\n"
            f"- Depressed: {row['PHQ_8Depressed']}\n"
            f"- Sleep: {row['PHQ_8Sleep']}\n"
            f"- Tired: {row['PHQ_8Tired']}\n"
            f"- Appetite: {row['PHQ_8Appetite']}\n"
            f"- Failure: {row['PHQ_8Failure']}\n"
            f"- Concentrating: {row['PHQ_8Concentrating']}\n"
            f"- Moving: {row['PHQ_8Moving']}\n"
            f"- Total: {row['PHQ_8Total']}\n"
            "Q: Explain the reasoning behind each score.\nA: Let's think step by step."
        )

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
            )

            rationale = response.choices[0].message.content.strip()

            demo_list.append({
                "question": user_prompt,
                "rationale": rationale,
                "scores": {
                    "No Interest": row['PHQ_8NoInterest'],
                    "Depressed": row['PHQ_8Depressed'],
                    "Sleep": row['PHQ_8Sleep'],
                    "Tired": row['PHQ_8Tired'],
                    "Appetite": row['PHQ_8Appetite'],
                    "Failure": row['PHQ_8Failure'],
                    "Concentrating": row['PHQ_8Concentrating'],
                    "Moving": row['PHQ_8Moving'],
                    "Total": row['PHQ_8Total']
                }
            })

        except Exception as e:
            print(f"[ERROR] GPT failed on PID {row['Participant_ID']} → {e}")
            continue

    # Save output
    with open(output_json_path, "w") as f:
        json.dump({"demo": demo_list}, f, indent=4)
    print(f"✅ Saved {len(demo_list)} demo rationales to: {output_json_path}")


def format_autocot_demo_text(json_path, output_csv_path=None):
    """
    Formats demo examples from GPT rationale output into Auto-CoT compatible Q&A format.
    
    Args:
        json_path (str): Path to the JSON file containing demo rationales.
        output_csv_path (str, optional): Path to save the formatted demos as a CSV file.
    
    Returns:
        list: A list of formatted demo strings.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    formatted_demos = []
    for example in data["demo"]:
        # Extract clean transcript
        transcript = example["question"].split("Transcript:", 1)[-1].split("True PHQ-8")[0].strip()
        rationale = example["rationale"].strip()
        scores = example["scores"]

        score_str = ", ".join(f"{k}: {v}" for k, v in scores.items())
        formatted = (
            "Q: You are a clinical psychologist. You read transcripts of a patient from a diagnostic interview and estimate the PHQ-8 score by reasoning through what the participant said. "
            "Evaluate on the following dimensions: PHQ_8NoInterest,PHQ_8Depressed,PHQ_8Sleep,PHQ_8Tired,PHQ_8Appetite,PHQ_8Failure,PHQ_8Concentrating,PHQ_8Moving,PHQ_8Total.\n"
            f"{transcript}\n"
            f"A: Let's think step by step. {rationale}. The answer is {score_str}"
        )
        formatted_demos.append(formatted)

    if output_csv_path:
        pd.DataFrame({"Formatted Demo": formatted_demos}).to_csv(output_csv_path, index=False)
        print(f"Formatted demos saved to {output_csv_path}")

    return formatted_demos


def run_autocot_inference(training_csv_path, formatted_demos, output_path, model="gpt-4o"):
    """
    Runs Auto-CoT inference on training data and saves predictions.
    
    Args:
        training_csv_path (str): Path to the training transcripts CSV file.
        formatted_demos (list): List of formatted demo strings.
        output_path (str): Path to save the predictions.
        model (str): Model name for GPT-4o.
    
    Outputs:
        Saves predictions to a CSV file with columns: Participant_ID, Prompt, Response, and PHQ-8 scores.
    """
    # Load the training transcripts
    df = pd.read_csv(training_csv_path)
    predictions = []

    # Iterate through each training transcript
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running Auto-CoT Inference"):
        participant_id = row["Participant_ID"]
        transcript_text = row["Text"]

        # Print progress info for tracking
        print(f"\n🔍 Processing Participant_ID: {participant_id} ({idx+1}/{len(df)})")

        # Construct the full prompt using demo rationales + the current transcript
        prompt = (
            "\n\n".join(formatted_demos) + "\n\n"
            "Q: You are a clinical psychologist. You read transcripts of a patient from a diagnostic interview and estimate the PHQ-8 score by reasoning through what the participant said. "
            "Evaluate on the following dimensions: PHQ_8NoInterest,PHQ_8Depressed,PHQ_8Sleep,PHQ_8Tired,PHQ_8Appetite,PHQ_8Failure,PHQ_8Concentrating,PHQ_8Moving,PHQ_8Total.\n"
            f"{transcript_text}\n"
            "A: Let's think step by step. After analyzing the transcript, you MUST output the scores in this exact format:\n"
            "PHQ_8NoInterest: [score]\n"
            "PHQ_8Depressed: [score]\n"
            "PHQ_8Sleep: [score]\n"
            "PHQ_8Tired: [score]\n"
            "PHQ_8Appetite: [score]\n"
            "PHQ_8Failure: [score]\n"
            "PHQ_8Concentrating: [score]\n"
            "PHQ_8Moving: [score]\n"
            "PHQ_8Total: [score]\n"
            "Again, make sure to output the scores in the above format."
        )

        try:
            # Query GPT-4o for reasoning and prediction
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip()

            # Extract individual scores from the response using regex
            # Look for patterns like "PHQ_8NoInterest: X"
            score_patterns = {
                "PHQ_8NoInterest": r"PHQ_8NoInterest:\s*(\d+)",
                "PHQ_8Depressed": r"PHQ_8Depressed:\s*(\d+)",
                "PHQ_8Sleep": r"PHQ_8Sleep:\s*(\d+)",
                "PHQ_8Tired": r"PHQ_8Tired:\s*(\d+)",
                "PHQ_8Appetite": r"PHQ_8Appetite:\s*(\d+)",
                "PHQ_8Failure": r"PHQ_8Failure:\s*(\d+)",
                "PHQ_8Concentrating": r"PHQ_8Concentrating:\s*(\d+)",
                "PHQ_8Moving": r"PHQ_8Moving:\s*(\d+)",
                "PHQ_8Total": r"PHQ_8Total:\s*(\d+)"
            }

            # Extract scores using patterns
            scores = {}
            for score_name, pattern in score_patterns.items():
                match = re.search(pattern, reply)
                scores[score_name] = float(match.group(1)) if match else None

            # Store prediction with participant ID
            predictions.append({
                "Participant_ID": participant_id,
                "Response": reply,
                **scores  # Unpack the scores dictionary
            })

        except Exception as e:
            # Handle GPT failure gracefully
            print(f"[ERROR] GPT failed on PID {participant_id} → {e}")
            predictions.append({
                "Participant_ID": participant_id,
                "Prompt": prompt,
                "Response": "",
                "PHQ_8NoInterest": None,
                "PHQ_8Depressed": None,
                "PHQ_8Sleep": None,
                "PHQ_8Tired": None,
                "PHQ_8Appetite": None,
                "PHQ_8Failure": None,
                "PHQ_8Concentrating": None,
                "PHQ_8Moving": None,
                "PHQ_8Total": None
            })

    # Save all predictions to a CSV file
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    print(f"\n✅ Finished! Saved predictions to: {output_path}")


def calculate_statistics(predictions_csv_path, training_csv_path):
    """
    Calculates MAE and RMSE for each PHQ-8 subscore and the total score.
    
    Args:
        predictions_csv_path (str): Path to the predictions CSV file.
        training_csv_path (str): Path to the training transcripts CSV file.
    
    Returns:
        tuple: Dictionaries containing MAE and RMSE for each score.
    """
    # Load the predictions and training CSVs
    predictions_df = pd.read_csv(predictions_csv_path)
    training_df = pd.read_csv(training_csv_path)
    
    # Define the score columns
    score_columns = [
        "PHQ_8NoInterest", "PHQ_8Depressed", "PHQ_8Sleep", "PHQ_8Tired",
        "PHQ_8Appetite", "PHQ_8Failure", "PHQ_8Concentrating", "PHQ_8Moving", "PHQ_8Total"
    ]
    
    # Initialize dictionaries to store MAE and RMSE
    mae_scores = {}
    rmse_scores = {}
    
    # Calculate MAE and RMSE for each score
    for column in score_columns:
        # Merge on Participant_ID to align gold and predicted scores
        merged_df = pd.merge(training_df[['Participant_ID', column]], predictions_df[['Participant_ID', column]], on='Participant_ID', suffixes=('_gold', '_pred'))
        
        gold_scores = merged_df[f'{column}_gold']
        predicted_scores = merged_df[f'{column}_pred']
        
        # Drop NaN values
        valid_indices = ~predicted_scores.isna()
        gold_scores = gold_scores[valid_indices]
        predicted_scores = predicted_scores[valid_indices]
        
        # Calculate MAE and RMSE
        mae = np.mean(np.abs(gold_scores - predicted_scores))
        rmse = np.sqrt(np.mean((gold_scores - predicted_scores) ** 2))
        
        # Store the results
        mae_scores[column] = mae
        rmse_scores[column] = rmse
        
        # Print statistics for each score
        print(f"{column} - MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    
    # Save MAE and RMSE scores to a file
    results_output_path = '/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT/outputs/statistics_results.txt'
    with open(results_output_path, 'w') as f:
        f.write("MAE Scores:\n")
        for score, value in mae_scores.items():
            f.write(f"{score}: {value:.3f}\n")
        f.write("\nRMSE Scores:\n")
        for score, value in rmse_scores.items():
            f.write(f"{score}: {value:.3f}\n")
    print(f"Results saved to {results_output_path}")
    
    return mae_scores, rmse_scores


# Main function to execute the Auto-CoT process
if __name__ == "__main__":
    # # Load detailed PHQ-8 labels
    # labels_path = "/Users/kevinawang/Documents/GitHub/VA-classifier/data/edaic_raw/labels/Detailed_PHQ8_Labels.csv"
    # labels_df = pd.read_csv(labels_path)
    # transcripts_path = "/Users/kevinawang/Documents/GitHub/VA-classifier/data/edaic_transcripts"

    # print("Calling merge_and_split_transcripts_and_labels...🫠")
    # merge_and_split_transcripts_and_labels(transcripts_path, labels_df)

    # # Generate rationale demos
    # demo_csv_path = '/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT/outputs/demo_transcripts.csv'
    output_json_path = '/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT/outputs/demo_rationales.json'
    # print("Calling generate_rationale_demos_with_gpt...🫠")
    # generate_rationale_demos_with_gpt(demo_csv_path, output_json_path)
    
    # # Format and save demos
    formatted_demos_csv_path = '/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT/outputs/formatted_demos.csv'
    # print("Calling format_autocot_demo_text...🫠")
    formatted_demos = format_autocot_demo_text(output_json_path, formatted_demos_csv_path)

    # # Run autoCOT and save predictions
    training_csv_path = "/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT/outputs/training_transcripts.csv"
    # training_csv_path = "/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT/outputs/training_transcripts_TEST.csv"
    output_predictions_path = "/Users/kevinawang/Documents/GitHub/VA-classifier/src/models/autoCOT/outputs/autocot_predictions_TEST.csv"
    print("Calling run_autocot_inference...🫠")
    run_autocot_inference(training_csv_path, formatted_demos, output_predictions_path)

    # Calculate statistics
    print("Calling calculate_statistics...🫠")
    mae_scores, rmse_scores = calculate_statistics(output_predictions_path, training_csv_path)