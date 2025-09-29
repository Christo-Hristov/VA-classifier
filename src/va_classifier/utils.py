import torch
from transformers import AutoTokenizer
from src.models.roberta_direct_to_VA_edits import VARegressor

def get_va_scores(texts, model_path=None, device=None):
    """
    Get valence-arousal scores for a list of input texts using the RoBERTa direct VA model.
    
    Args:
        texts (list): List of input text strings
        model_path (str, optional): Path to saved model weights. If None, uses default path
        device (str, optional): Device to run model on ('cuda' or 'cpu'). If None, auto-detects
        
    Returns:
        list: List of [valence, arousal] scores for each input text
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and tokenizer
    model = VARegressor(n_layers=3, hidden_fc1=256, hidden_fc2=128, hidden_fc3=64)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Load model weights if path provided
    if model_path is None:
        model_path = "src/models/roberta_emobank/best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    # Process texts in batches
    batch_size = 32
    all_scores = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encodings = tokenizer(batch_texts, 
                            padding=True, 
                            truncation=True, 
                            max_length=512,
                            return_tensors="pt")
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        # Get predictions
        with torch.no_grad():
            scores = model(input_ids, attention_mask)
            scores = scores.cpu().numpy()
            
        all_scores.extend(scores.tolist())
    
    return all_scores