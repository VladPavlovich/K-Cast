import torch
import random
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Import our modules
from dataset_utils import parse_uuid_dataset
from activations import ActivationExtractor
from kcast import KCASTDatastore, KCASTSteerer

# ================= Configuration =================
MODEL_NAME = "/home/nlp-shared/akash_models/Qwen2.5_7B"
LAYER_IDX = -14
K_NEIGHBORS = 32
TRAIN_FILE = "train_parsed.json" # Your UUID JSON file
VAL_FILE = "val_parsed.json"     # Your UUID JSON file
# =================================================

def main():
    # 1. Load Model (ONCE)
    extractor = ActivationExtractor(MODEL_NAME, layer_idx=LAYER_IDX)
    
    # 2. Parse Data
    print("Parsing datasets...")
    train_data = parse_uuid_dataset(TRAIN_FILE)
    val_data = parse_uuid_dataset(VAL_FILE)
    
    # 3. Build Datastore & Compute Delta
    # We do this in one pass to avoid running the model twice
    print("Extracting activations and building datastore...")
    datastore = KCASTDatastore(device=extractor.device)
    
    # We store valid/invalid vectors separately for Delta calculation
    valid_vecs = []
    invalid_vecs = []

    for i, item in enumerate(train_data):
        # Run forward pass
        phi = extractor.get_phi(item["text"])
        
        # Add to datastore
        datastore.add(phi, item["label"])
        
        # Sort for Delta calculation
        if item["label"] == "valid":
            valid_vecs.append(phi)
        else:
            invalid_vecs.append(phi)
            
        if i % 100 == 0: print(f"  Processed {i}/{len(train_data)}")

    datastore.finalize()
    
    # 4. Compute Steering Vector (Delta)
    print("Computing Delta (Mean Valid - Mean Invalid)...")
    mu_valid = torch.stack(valid_vecs).mean(dim=0)
    mu_invalid = torch.stack(invalid_vecs).mean(dim=0)
    delta_c = mu_valid - mu_invalid
    
    # 5. Steering Evaluation Loop
    #alpha_list = [random.uniform(-3.0, 3.0) for _ in range(5)] # Reduced range for testing
    alpha_list = [0.0]
    results_log = {}

    print(f"\nStarting Sweep over Alphas: {alpha_list}")
    
    # We need to access the internal pytorch layer to register the hook
    target_layer = extractor.model.model.layers[LAYER_IDX]

    for alpha in alpha_list:
        print(f"\n--- Testing Alpha: {alpha:.4f} ---")
        
        # Register the Steering Hook
        steerer = KCASTSteerer(delta_c, datastore, alpha=alpha, k=K_NEIGHBORS)
        hook_handle = target_layer.register_forward_hook(steerer)
        
        # Evaluate on Validation Set
        preds = []
        golds = []
        
        # Simple valid/invalid token check
        valid_token_id = extractor.tokenizer.encode("valid", add_special_tokens=False)[0]
        invalid_token_id = extractor.tokenizer.encode("invalid", add_special_tokens=False)[0]

        for item in val_data:
            inputs = extractor.tokenizer(item["text"], return_tensors="pt").to(extractor.device)
            
            with torch.no_grad():
                out = extractor.model(**inputs)
            
            # Simple Logit check for "valid" vs "invalid"
            logits = out.logits[0, -1, :]
            score_valid = logits[valid_token_id].item()
            score_invalid = logits[invalid_token_id].item()
            
            prediction = "valid" if score_valid > score_invalid else "invalid"
            
            preds.append(prediction)
            golds.append(item["label"])

        # Calculate Accuracy
        acc = accuracy_score(golds, preds)
        print(f"Accuracy: {acc:.4f}")
        results_log[str(alpha)] = acc
        
        # CRITICAL: Remove hook before next alpha!
        hook_handle.remove()

    # Save results
    with open("results_kcast.json", "w") as f:
        json.dump(results_log, f, indent=4)
    print("\nDone. Results saved to results_kcast.json")

if __name__ == "__main__":
    main()