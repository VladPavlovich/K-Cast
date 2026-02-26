import os
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
from dataset_utils import parse_uuid_dataset
from activations import ActivationExtractor

# --- KCAST CORE COMPONENTS ---

class KCASTDatastore:
    def __init__(self, device="cuda"):
        self.phis = []
        self.labels = []
        self.device = device

    def add(self, phi, label):
        self.phis.append(phi.to(self.device))
        self.labels.append(1 if label == "valid" else -1)

    def finalize(self):
        if not self.phis:
            return
        self.phis_tensor = torch.stack(self.phis)
        self.phis_norm = F.normalize(self.phis_tensor, p=2, dim=1)
        self.labels_tensor = torch.tensor(self.labels, device=self.device)

    def query_knn(self, phi_x, k=32):
        phi_x = phi_x.to(self.device)
        phi_x_norm = F.normalize(phi_x.unsqueeze(0), p=2, dim=1)
        sims = torch.mm(phi_x_norm, self.phis_norm.t()).squeeze()
        k = min(k, len(self.labels))
        topk_indices = torch.topk(sims, k).indices
        neighbor_labels = self.labels_tensor[topk_indices]
        return 1 if neighbor_labels.sum() >= 0 else -1

class KCASTSteerer:
    def __init__(self, delta_base, datastore, alpha=1.0, k=32):
        self.delta_base = delta_base
        self.datastore = datastore
        self.alpha = alpha
        self.k = k

    def __call__(self, module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Steering the last token position
        last_token_vec = hidden_states[0, -1, :].detach()
        y_hat = self.datastore.query_knn(last_token_vec, k=self.k)
        
        delta = self.delta_base.to(hidden_states.device).to(hidden_states.dtype)
        new_hidden = hidden_states.clone()
        
        # Formula: phi_new = phi - (y_hat * alpha * delta)
        update = y_hat * self.alpha * delta
        new_hidden[:, -1, :] -= update

        return (new_hidden,) if isinstance(output, tuple) else new_hidden

# --- EXECUTION SCRIPT ---

MODEL_PATH = "/home/nlp-shared/akash_models/fine-tuned_models/syllogistic_reasoning/model-weights-qlora-syllogisms/Qwen2.5_7B/merged"
TEST_DATA_PATH = "test_data_subtask_1.json"
OUTPUT_FILE = "predictions_kcast.json"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# Hyperparameters
LAYER_IDX = -14
ALPHA = 5.0 
K_NEIGHBORS = 16
MAX_NEW_TOKENS = 256
SYSTEM_MSG = "Take a deep breath and go step by step. Reason through the logic then output 'valid' or 'invalid'."

def generate_with_kcast(extractor, input_ids, max_new_tokens=256):
    generated = input_ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = extractor.model(generated)
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if next_token.item() == extractor.tokenizer.eos_token_id:
                break
    return extractor.tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    extractor = ActivationExtractor(MODEL_PATH, layer_idx=LAYER_IDX)
    
    # 1. Prepare Datastore and Delta Vector
    print("Loading data and building KCAST Datastore...")
    train_data = parse_uuid_dataset("train_parsed.json")
    val_data = parse_uuid_dataset("val_parsed.json")
    combined_data = train_data + val_data
    
    datastore = KCASTDatastore(device=extractor.device)
    valid_vecs, invalid_vecs = [], []
    
    # Use a subset to build the datastore and delta
    for item in tqdm(combined_data[:600], desc="Extracting Activations"):
        phi = extractor.get_phi(item["text"])
        datastore.add(phi, item["label"])
        if item["label"] == "valid":
            valid_vecs.append(phi)
        else:
            invalid_vecs.append(phi)
            
    datastore.finalize()
    delta_base = torch.stack(valid_vecs).mean(0) - torch.stack(invalid_vecs).mean(0)

    # 2. Setup Steerer Hook
    steerer = KCASTSteerer(delta_base, datastore, alpha=ALPHA, k=K_NEIGHBORS)
    target_layer = extractor.model.model.layers[LAYER_IDX]
    hook_handle = target_layer.register_forward_hook(steerer)

    # 3. Inference on Test Data
    with open(TEST_DATA_PATH, "r") as f:
        test_items = json.load(f)

    final_results = []
    print(f"Running KCAST Inference on {len(test_items)} items...")

    for item in tqdm(test_items):
        prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{item['syllogism']}\nDecision:<|im_end|>\n<|im_start|>assistant\n"
        
        input_ids = extractor.tokenizer(prompt, return_tensors="pt").input_ids.to(extractor.device)
        full_text = generate_with_kcast(extractor, input_ids, MAX_NEW_TOKENS)
        
        # Parsing decision from assistant response
        decision_area = full_text.split("Decision:")[-1].lower()
        if "invalid" in decision_area:
            prediction = False
        elif "valid" in decision_area:
            prediction = True
        else:
            # Final fallback
            prediction = "invalid" not in full_text.lower() and "valid" in full_text.lower()

        final_results.append({
            "id": item["id"],
            "validity": prediction
        })

    # 4. Save and Cleanup
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_results, f, indent=4)

    hook_handle.remove()
    print(f"Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()