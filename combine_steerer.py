import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataset_utils import parse_uuid_dataset
from activations import ActivationExtractor

# --- CONFIGURATION ---
MODEL_PATH = "/home/nlp-shared/akash_models/fine-tuned_models/syllogistic_reasoning/model-weights-qlora-syllogisms/Qwen2.5_7B/merged"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# Define the search space
LAYERS_TO_TEST = [ -14] # Late to mid layers
ALPHA_VALUES = [2,4,6,8,10,12,14,18]      # Steering magnitudes
GAMMA = 1.5                           # Fixed DCM contrast factor

class DCMSteerer:
    def __init__(self, delta_vector, alpha=5.0, gamma=2.0):
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
        update = self.alpha * self.delta.to(hidden_states.device).to(hidden_states.dtype)
        new_hidden[:, -1, :] += update
        return (new_hidden,) if isinstance(output, tuple) else new_hidden

def get_dcm_logits(extractor, inputs, steerer):
    """Executes the Contrastive Double-Pass."""
    steerer.active = False
    with torch.no_grad():
        out_base = extractor.model(**inputs)
        logits_base = out_base.logits[0, -1, :]

    steerer.active = True
    with torch.no_grad():
        out_steer = extractor.model(**inputs)
        logits_steer = out_steer.logits[0, -1, :]

    logits_final = logits_steer + steerer.gamma * (logits_steer - logits_base)
    return logits_final

def main():
    # Load model and data once
    extractor = ActivationExtractor(MODEL_PATH, layer_idx=LAYERS_TO_TEST[0])
    train_data = parse_uuid_dataset("train_parsed.json")
    val_data = parse_uuid_dataset("val_parsed.json")

    valid_id = extractor.tokenizer.encode("valid", add_special_tokens=False)[0]
    invalid_id = extractor.tokenizer.encode("invalid", add_special_tokens=False)[0]
    
    SYSTEM_MSG = "Take a deep breath and go step by step. Reason through the logic then output 'valid' or 'invalid'."
    results_table = []


    #base accuracy
    print("Computing Baseline Accuracy (No Steering)...")
    preds, golds = [], []
    for item in tqdm(val_data):
        prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{item['text']}\nDecision:<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.device)
        with torch.no_grad():
            out = extractor.model(**inputs)
        
        logits = out.logits[0, -1, :]
        v_score = logits[valid_id].item()
        inv_score = logits[invalid_id].item()
        
        preds.append("valid" if v_score > inv_score else "invalid")
        golds.append(item["label"])
    base_acc = accuracy_score(golds, preds)
    print(f"Baseline Accuracy: {base_acc:.4f}")



    for lay_idx in LAYERS_TO_TEST:
        print(f"\n{'='*20} TESTING LAYER {lay_idx} {'='*20}")
        
        # 1. Update extractor layer and re-compute DoM vector for this depth
        extractor.layer_idx = lay_idx
        valid_vecs = [extractor.get_phi(i["text"]) for i in train_data[:200] if i["label"] == "valid"]
        invalid_vecs = [extractor.get_phi(i["text"]) for i in train_data[:200] if i["label"] == "invalid"]
        delta_layer = torch.stack(valid_vecs).mean(0) - torch.stack(invalid_vecs).mean(0)

        # 2. Register hook for this layer
        steerer = DCMSteerer(delta_layer, gamma=GAMMA)
        target_layer = extractor.model.model.layers[lay_idx]
        hook_handle = target_layer.register_forward_hook(steerer.hook_fn)

        for alpha in ALPHA_VALUES:
            steerer.alpha = alpha
            preds, golds = [], []
            margins = []

            print(f"Running Alpha {alpha}...")
            for item in tqdm(val_data):
                prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
                prompt += f"<|im_start|>user\n{item['text']}\nDecision:<|im_end|>\n<|im_start|>assistant\n"
                
                inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.device)
                logits = get_dcm_logits(extractor, inputs, steerer)
                
                v_score, inv_score = logits[valid_id].item(), logits[invalid_id].item()
                preds.append("valid" if v_score > inv_score else "invalid")
                golds.append(item["label"])
                margins.append(v_score - inv_score)

            acc = accuracy_score(golds, preds)
            avg_margin = np.mean(margins)
            results_table.append((lay_idx, alpha, acc, avg_margin))
            print(f"Layer {lay_idx} | Alpha {alpha} | Acc: {acc:.4f} | Avg Margin: {avg_margin:+.2f}")

        # Remove hook before switching layers
        hook_handle.remove()
        torch.cuda.empty_cache()

    # --- FINAL SUMMARY ---
    print("\n" + "#"*40)
    print(f"{'Layer':<8} | {'Alpha':<8} | {'Accuracy':<10} | {'Margin':<10}")
    print("-" * 45)
    for l, a, acc, m in results_table:
        print(f"{l:<8} | {a:<8} | {acc:<10.4f} | {m:<+10.2f}")
    print("#"*40)

if __name__ == "__main__":
    main()