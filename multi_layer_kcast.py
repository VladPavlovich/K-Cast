import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataset_utils import parse_uuid_dataset
from activations import ActivationExtractor

# --- CONFIGURATION ---
#MODEL_PATH = "/home/nlp-shared/akash_models/fine-tuned_models/syllogistic_reasoning/model-weights-qlora-syllogisms/Qwen2.5_7B/merged"
MODEL_PATH = "/home/nlp-shared/akash_models/Qwen2.5_7B"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# Multi-layer block (5 layers)
BLOCK_LAYERS = [-12, -13, -14, -15, -16] 
# Alpha sweep: testing intensities across the block
ALPHA_SWEEP = [0.0,-3.0,-2.0,-1.0,3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
K_VAL = 24
SYSTEM_MSG = "Take a deep breath and go step by step. Reason through the logic then output 'valid' or 'invalid'."

class MultiLayerKCAST:
    def __init__(self, layer_ids, device="cuda"):
        self.layer_ids = layer_ids
        self.device = device
        self.datastores = {l: {"phis": [], "labels": []} for l in layer_ids}
        self.deltas = {}
        self.finalized = False
        self.current_alpha = 1.0

    def add_activation(self, layer_id, phi, label):
        self.datastores[layer_id]["phis"].append(phi.detach().to(self.device))
        self.datastores[layer_id]["labels"].append(1 if label == "valid" else -1)

    def finalize(self):
        for l in self.layer_ids:
            phis = torch.stack(self.datastores[l]["phis"])
            self.datastores[l]["phis_norm"] = F.normalize(phis, p=2, dim=1)
            self.datastores[l]["labels_tensor"] = torch.tensor(self.datastores[l]["labels"], device=self.device)
            
            valid_mask = self.datastores[l]["labels_tensor"] == 1
            invalid_mask = self.datastores[l]["labels_tensor"] == -1
            self.deltas[l] = phis[valid_mask].mean(0) - phis[invalid_mask].mean(0)
        self.finalized = True

    def get_steer_fn(self, layer_id):
        def hook_fn(module, inputs, output):
            if not self.finalized: return output
            h = output[0] if isinstance(output, tuple) else output
            last_token = h[0, -1, :].detach()
            
            # Query layer-specific datastore
            query_norm = F.normalize(last_token.unsqueeze(0), p=2, dim=1)
            sims = torch.mm(query_norm, self.datastores[layer_id]["phis_norm"].t()).squeeze()
            topk_idx = torch.topk(sims, min(K_VAL, len(sims))).indices
            y_hat = 1 if self.datastores[layer_id]["labels_tensor"][topk_idx].sum() >= 0 else -1
            
            new_h = h.clone()
            # Update: phi = phi - (y_hat * alpha * delta)
            update = y_hat * self.current_alpha * self.deltas[layer_id].to(h.device).to(h.dtype)
            new_h[:, -1, :] -= update
            return (new_h,) if isinstance(output, tuple) else new_h
        return hook_fn

def main():
    # 1. Load Model and Data
    extractor = ActivationExtractor(MODEL_PATH, layer_idx=BLOCK_LAYERS[0])
    train_data = parse_uuid_dataset("train_parsed.json")
    val_data = parse_uuid_dataset("val_parsed.json")
    
    # 2. Build the Multi-Layer Datastore
    kcast_system = MultiLayerKCAST(BLOCK_LAYERS, device=extractor.device)
    print(f"--- Training KCAST Datastores (Layers: {BLOCK_LAYERS}) ---")
    for item in tqdm(train_data[:400], desc="Extracting Train Activations"):
        for l_idx in BLOCK_LAYERS:
            extractor.layer_idx = l_idx
            phi = extractor.get_phi(item["text"])
            kcast_system.add_activation(l_idx, phi, item["label"])
    kcast_system.finalize()

    # 3. Register Hooks
    handles = []
    for l_idx in BLOCK_LAYERS:
        target_layer = extractor.model.model.layers[l_idx]
        handles.append(target_layer.register_forward_hook(kcast_system.get_steer_fn(l_idx)))

    # 4. Token IDs for evaluation
    valid_id = extractor.tokenizer.encode("valid", add_special_tokens=False)[0]
    invalid_id = extractor.tokenizer.encode("invalid", add_special_tokens=False)[0]

    # 5. Validation Sweep
    print(f"\n--- Starting Alpha Sweep on Validation Set ({len(val_data)} items) ---")
    results = []

    for alpha in ALPHA_SWEEP:
        kcast_system.current_alpha = alpha
        preds, golds = [], []
        
        for item in tqdm(val_data, desc=f"Val Accuracy [Alpha={alpha}]"):
            prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n<|im_start|>user\n{item['text']}\nDecision:<|im_end|>\n<|im_start|>assistant\n"
            inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.device)
            
            with torch.no_grad():
                out = extractor.model(**inputs)
                logits = out.logits[0, -1, :]
                v_score = logits[valid_id].item()
                inv_score = logits[invalid_id].item()
                
            preds.append("valid" if v_score > inv_score else "invalid")
            golds.append(item["label"])
        
        acc = accuracy_score(golds, preds)
        results.append((alpha, acc))
        print(f"RESULT: Alpha {alpha} | Accuracy: {acc:.4f}")

    # Final Summary Table
    print("\n" + "="*30)
    print(f"{'Alpha':<10} | {'Val Accuracy':<10}")
    print("-" * 25)
    for a, acc in results:
        marker = " <--- BEST" if acc == max(r[1] for r in results) else ""
        print(f"{a:<10} | {acc:<12.4f}{marker}")
    print("="*30)

    # Cleanup
    for h in handles: h.remove()

if __name__ == "__main__":
    main()