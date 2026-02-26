import os
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataset_utils import parse_uuid_dataset
from activations import ActivationExtractor

# --- CONFIGURATION ---
MODEL_PATH = "/home/nlp-shared/akash_models/fine-tuned_models/syllogistic_reasoning/model-weights-qlora-syllogisms/Qwen2.5_7B/merged"
TEST_DATA_PATH = "test_data_subtask_1.json"  # The file with IDs and Syllogisms
OUTPUT_FILE = "dcm_predictions.json"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# Optimal Params (Based on your search space)
BEST_LAYER = -14
BEST_ALPHA = 18
GAMMA = 1.5
SYSTEM_MSG = "Take a deep breath and go step by step. Reason through the logic then output 'valid' or 'invalid'."

class DCMSteerer:
    def __init__(self, delta_vector, alpha=5.0, gamma=1.5):
        self.delta = delta_vector / (torch.norm(delta_vector) + 1e-6)
        self.alpha = alpha
        self.gamma = gamma
        self.active = False 

    def hook_fn(self, module, inputs, output):
        if not self.active:
            return output
        
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        new_hidden = hidden_states.clone()
        # Steering the last token position
        update = self.alpha * self.delta.to(hidden_states.device).to(hidden_states.dtype)
        new_hidden[:, -1, :] += update
        return (new_hidden,) if isinstance(output, tuple) else new_hidden

def get_dcm_logits(extractor, inputs, steerer):
    """Executes the Contrastive Double-Pass."""
    # Pass 1: Base Model
    steerer.active = False
    with torch.no_grad():
        out_base = extractor.model(**inputs)
        logits_base = out_base.logits[0, -1, :]

    # Pass 2: Steered Model
    steerer.active = True
    with torch.no_grad():
        out_steer = extractor.model(**inputs)
        logits_steer = out_steer.logits[0, -1, :]

    # DCM Formula: Logits = Steered + Gamma * (Steered - Base)
    logits_final = logits_steer + steerer.gamma * (logits_steer - logits_base)
    return logits_final

def main():
    # 1. Initialize Model
    extractor = ActivationExtractor(MODEL_PATH, layer_idx=BEST_LAYER)
    
    # 2. Compute Steering Vector using both Train and Val
    print("Computing Steering Vector from combined Train/Val data...")
    train_data = parse_uuid_dataset("train_parsed.json")
    val_data = parse_uuid_dataset("val_parsed.json")
    combined_data = train_data + val_data

    valid_vecs = [extractor.get_phi(i["text"]) for i in combined_data if i["label"] == "valid"]
    invalid_vecs = [extractor.get_phi(i["text"]) for i in combined_data if i["label"] == "invalid"]
    
    delta_vector = torch.stack(valid_vecs).mean(0) - torch.stack(invalid_vecs).mean(0)

    # 3. Setup Steerer and Hook
    steerer = DCMSteerer(delta_vector, alpha=BEST_ALPHA, gamma=GAMMA)
    target_layer = extractor.model.model.layers[BEST_LAYER]
    hook_handle = target_layer.register_forward_hook(steerer.hook_fn)

    # 4. Load Test Data for Inference
    with open(TEST_DATA_PATH, "r") as f:
        test_items = json.load(f)

    valid_id = extractor.tokenizer.encode("valid", add_special_tokens=False)[0]
    invalid_id = extractor.tokenizer.encode("invalid", add_special_tokens=False)[0]

    final_results = []

    print(f"Running Inference on {len(test_items)} items...")
    for item in tqdm(test_items):
        # Format prompt to match training style
        prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{item['syllogism']}\nDecision:<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.device)
        
        # Get contrastive logits
        logits = get_dcm_logits(extractor, inputs, steerer)
        
        v_score = logits[valid_id].item()
        inv_score = logits[invalid_id].item()
        
        # Determine validity
        is_valid = v_score > inv_score
        
        final_results.append({
            "id": item["id"],
            "validity": is_valid
        })

    # 5. Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"Inference complete. Results saved to {OUTPUT_FILE}")
    hook_handle.remove()

if __name__ == "__main__":
    main()