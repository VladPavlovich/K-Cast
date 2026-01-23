import os
import torch
import json
import random
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Assuming these are your utility files
from dataset_utils import parse_uuid_dataset
from activations import ActivationExtractor
from riser import RiserRouter, RiserSteerer, train_riser

# Set GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

MODEL_CONFIGS = {
    "Qwen2.5_7B": {
        "path": "/home/nlp-shared/akash_models/fine-tuned_models/syllogistic_reasoning/model-weights-qlora-syllogisms/Qwen2.5_7B/merged",
        "layers": [-14] # Added a later layer for comparison
    }
}

def evaluate_with_metrics(extractor, val_data, valid_id, invalid_id):
    """Evaluates the model and returns accuracy and the average logit margin."""
    preds, golds = [], []
    margins = []
    
    for item in val_data:
        inputs = extractor.tokenizer(item["text"], return_tensors="pt").to(extractor.device)
        with torch.no_grad():
            out = extractor.model(**inputs)
        
        logits = out.logits[0, -1, :]
        v_score = logits[valid_id].item()
        inv_score = logits[invalid_id].item()
        
        # Margin = Valid Logit - Invalid Logit
        # Positive means the model prefers 'valid'
        margins.append(v_score - inv_score)
        
        preds.append("valid" if v_score > inv_score else "invalid")
        golds.append(item["label"])
        
    acc = accuracy_score(golds, preds)
    avg_margin = np.mean(margins)
    return acc, avg_margin

def main():
    for key, config in MODEL_CONFIGS.items():
        for lay_idx in config["layers"]:
            print(f"\n" + "="*50)
            print(f"INITIALIZING: {key} | LAYER: {lay_idx}")
            print("="*50)
            
            extractor = ActivationExtractor(config["path"], layer_idx=lay_idx)
            
            # 1. Load Data
            train_data = parse_uuid_dataset("train_parsed.json")
            val_data = parse_uuid_dataset("val_parsed.json")
            
            valid_id = extractor.tokenizer.encode("valid", add_special_tokens=False)[0]
            invalid_id = extractor.tokenizer.encode("invalid", add_special_tokens=False)[0]

            # 2. GET BASELINE (No Steering)
            print("\nCalculating Baseline Performance...")
            base_acc, base_margin = evaluate_with_metrics(extractor, val_data, valid_id, invalid_id)
            print(f"BASELINE -> Accuracy: {base_acc:.4f} | Avg Margin: {base_margin:.4f}")
            
            # 3. Compute Directional Vector (Delta)
            print("\nComputing Delta Base (Valid - Invalid)...")
            valid_vecs, invalid_vecs = [], []
            # Using a larger sample for a more stable delta
            for item in train_data[:400]: 
                phi = extractor.get_phi(item["text"])
                if item["label"] == "valid":
                    valid_vecs.append(phi)
                else:
                    invalid_vecs.append(phi)
            
            delta_c = torch.stack(valid_vecs).mean(0) - torch.stack(invalid_vecs).mean(0)
            
            # 4. Train the Dynamic Router
            # We pass the full train_data (or a balanced subset)
            router = train_riser(extractor, delta_c, train_data[:600])
            
            # 5. Sweep Alpha (Global Intensity)
            target_layer = extractor.model.model.layers[lay_idx]
            
            print(f"\n--- Starting Steering Sweep ---")
            results = []
            for alpha in [0.0, 2.0, 5.0, 10.0, 20.0]:
                steerer = RiserSteerer(delta_c, router, global_alpha=alpha)
                hook = target_layer.register_forward_hook(steerer)
                
                acc, margin = evaluate_with_metrics(extractor, val_data, valid_id, invalid_id)
                results.append((alpha, acc, margin))
                
                print(f"ALPHA: {alpha:<4} | ACC: {acc:.4f} | MARGIN: {margin:+.4f}")
                hook.remove()

            # Comparison Logic
            best_alpha = max(results, key=lambda x: x[1])
            print(f"\nBest Alpha for Layer {lay_idx}: {best_alpha[0]} with {best_alpha[1]:.4f} Accuracy")

            # Cleanup
            del extractor
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()