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

BLOCK_LAYERS = [-12, -13, -14, -15, -16] 
ALPHA_SWEEP = [3.0, 5.0, 7.0, 9.0] # DCM often handles higher alpha better than KCAST
GAMMA = 1.5                        # Contrastive amplification factor
SYSTEM_MSG = "Take a deep breath and go step by step. Reason through the logic then output 'valid' or 'invalid'."

class MultiLayerDCMSteerer:
    def __init__(self, layer_deltas, alpha=5.0):
        self.layer_deltas = layer_deltas # Dict: {layer_idx: delta_vector}
        self.alpha = alpha
        self.active = False 

    def get_hook(self, layer_idx):
        def hook_fn(module, inputs, output):
            if not self.active:
                return output
            
            h = output[0] if isinstance(output, tuple) else output
            # Normalize and move delta to correct device/dtype
            delta = self.layer_deltas[layer_idx]
            delta = delta / (torch.norm(delta) + 1e-6)
            
            new_h = h.clone()
            # Apply steering to the last token position
            update = self.alpha * delta.to(h.device).to(h.dtype)
            new_h[:, -1, :] += update
            
            return (new_h,) if isinstance(output, tuple) else new_h
        return hook_fn

def get_multi_layer_dcm_logits(extractor, inputs, steerer, gamma):
    """Executes the Contrastive Double-Pass with multi-layer influence."""
    # Pass 1: Base Model (No hooks active)
    steerer.active = False
    with torch.no_grad():
        out_base = extractor.model(**inputs)
        logits_base = out_base.logits[0, -1, :]

    # Pass 2: Steered Model (All hooks in the block active)
    steerer.active = True
    with torch.no_grad():
        out_steer = extractor.model(**inputs)
        logits_steer = out_steer.logits[0, -1, :]

    # DCM Formula: Final = Steered + Gamma * (Steered - Base)
    logits_final = logits_steer + gamma * (logits_steer - logits_base)
    return logits_final

def main():
    extractor = ActivationExtractor(MODEL_PATH, layer_idx=BLOCK_LAYERS[0])
    train_data = parse_uuid_dataset("train_parsed.json")
    val_data = parse_uuid_dataset("val_parsed.json")

    valid_id = extractor.tokenizer.encode("valid", add_special_tokens=False)[0]
    invalid_id = extractor.tokenizer.encode("invalid", add_special_tokens=False)[0]

    # 1. BASELINE CALCULATION
    print("\nCalculating Baseline Accuracy...")
    preds_b, golds_b = [], []
    for item in tqdm(val_data, desc="Baseline"):
        prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n<|im_start|>user\n{item['text']}\nDecision:<|im_end|>\n<|im_start|>assistant\n"
        inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.device)
        with torch.no_grad():
            out = extractor.model(**inputs)
        logits = out.logits[0, -1, :]
        preds_b.append("valid" if logits[valid_id] > logits[invalid_id] else "invalid")
        golds_b.append(item["label"])
    base_acc = accuracy_score(golds_b, preds_b)
    print(f"Baseline Accuracy: {base_acc:.4f}\n")

    # 2. COMPUTE LAYER-SPECIFIC DELTAS
    layer_deltas = {}
    print(f"Computing Deltas for Block: {BLOCK_LAYERS}")
    for l_idx in BLOCK_LAYERS:
        extractor.layer_idx = l_idx
        v_vecs = [extractor.get_phi(i["text"]) for i in train_data[:300] if i["label"] == "valid"]
        inv_vecs = [extractor.get_phi(i["text"]) for i in train_data[:300] if i["label"] == "invalid"]
        layer_deltas[l_idx] = torch.stack(v_vecs).mean(0) - torch.stack(inv_vecs).mean(0)

    # 3. INITIALIZE STEERER AND REGISTER BLOCK HOOKS
    steerer = MultiLayerDCMSteerer(layer_deltas)
    handles = []
    for l_idx in BLOCK_LAYERS:
        target_layer = extractor.model.model.layers[l_idx]
        handles.append(target_layer.register_forward_hook(steerer.get_hook(l_idx)))

    # 4. ALPHA SWEEP
    results = [(0.0, base_acc)]
    print(f"--- Starting Multi-Layer DCM Sweep (Gamma={GAMMA}) ---")
    
    for alpha in ALPHA_SWEEP:
        steerer.alpha = alpha
        preds, golds = [], []
        
        for item in tqdm(val_data, desc=f"DCM Alpha {alpha}"):
            prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n<|im_start|>user\n{item['text']}\nDecision:<|im_end|>\n<|im_start|>assistant\n"
            inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.device)
            
            # Use the contrastive pass
            logits = get_multi_layer_dcm_logits(extractor, inputs, steerer, GAMMA)
            
            preds.append("valid" if logits[valid_id] > logits[invalid_id] else "invalid")
            golds.append(item["label"])
        
        acc = accuracy_score(golds, preds)
        results.append((alpha, acc))
        print(f"RESULT: Alpha {alpha} | Accuracy: {acc:.4f}")

    # FINAL SUMMARY
    print("\n" + "="*35)
    print(f"{'Alpha':<10} | {'Val Accuracy':<10}")
    print("-" * 30)
    results.sort(key=lambda x: x[0])
    for a, acc in results:
        marker = " (Baseline)" if a == 0.0 else (" <--- BEST" if acc == max(r[1] for r in results) else "")
        print(f"{a:<10} | {acc:<12.4f}{marker}")
    print("="*35)

    for h in handles: h.remove()

if __name__ == "__main__":
    main()