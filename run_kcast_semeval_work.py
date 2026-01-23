import torch
import random
import json
from sklearn.metrics import accuracy_score

# Import our modules
from dataset_utils import parse_uuid_dataset
from activations import ActivationExtractor
from kcast import KCASTDatastore, KCASTSteerer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

model_path_dict =  {
        #"Llama_3.1_8B": "/home/nlp-shared/akash_models/Llama3.1_8B",
        #"Llama_3.2_3B": "/home/nlp-shared/akash_models/Llama3.2_3B",
        #"Qwen2.5_3B": "/home/nlp-shared/akash_models/Qwen2.5_3B",
        "Qwen2.5_7B": "/home/nlp-shared/akash_models/Qwen2.5_7B",
       #"Phi_4_4B": "/home/nlp-shared/akash_models/Phi_4_4B",
        #"Phi_4_15B": "/home/nlp-shared/akash_models/Phi_4_15B"
        }

model_params_dict = {
    "Llama_3.1_8B": {
        "layer_idx": [-10, -8, -6],
        "k_neighbors": 32
    },
    "Llama_3.2_3B": {
        "layer_idx": [-12, -10, -6],
        "k_neighbors": 32
    },
    "Qwen2.5_3B": {
        "layer_idx": [-14, -11, -7],
        "k_neighbors": 32
    },
    "Qwen2.5_7B": {
        "layer_idx": [-18, -14, -10],
        "k_neighbors": 32
    },
    "Phi_4_4B": {
        "layer_idx": [-14, -12, -10],
        "k_neighbors": 32
    },
    "Phi_4_15B": {
        "layer_idx": [-14, -12, -10],
        "k_neighbors": 32
    }
}

layer = 0
alpha = 0.0
max_scores_per_model = {
    "Llama_3.1_8B": (0,layer, alpha),
    "Llama_3.2_3B": (0,layer, alpha),
    "Qwen2.5_3B": (0,layer, alpha),
    "Qwen2.5_7B": (0,layer, alpha),
    "Phi_4_4B": (0,layer, alpha),
    "Phi_4_15B": (0,layer, alpha),
}


def main(MODEL_NAME, LAYER_IDX, K_NEIGHBORS):
    for lay_idx in LAYER_IDX:
        # 1. Load Model (ONCE)
        extractor = ActivationExtractor(MODEL_NAME, layer_idx=lay_idx)
        
        # 2. Parse Data
        print("Parsing datasets...")
        train_data = parse_uuid_dataset(TRAIN_FILE)
        val_data = parse_uuid_dataset(VAL_FILE)
        test_set = parse_uuid_dataset(TEST_FILE)
        
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
        alpha_list = [random.uniform(-3.0, 3.0) for _ in range(3)] # Reduced range for testing
        #alpha_list = [0.0]
        results_log = {}

        print(f"\nStarting Sweep over Alphas: {alpha_list}")
        
        # We need to access the internal pytorch layer to register the hook
        target_layer = extractor.model.model.layers[lay_idx]

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
            
            val_final_eval = {}
            for item in val_data:
                inputs = extractor.tokenizer(item["text"], return_tensors="pt").to(extractor.device)
                
                with torch.no_grad():
                    out = extractor.model(**inputs)
                
                # Simple Logit check for "valid" vs "invalid"
                logits = out.logits[0, -1, :]
                score_valid = logits[valid_token_id].item()
                score_invalid = logits[invalid_token_id].item()
                
                prediction = "valid" if score_valid > score_invalid else "invalid"
                
                #val_final_eval[item['id']] = {"Validity": True if score_valid > score_invalid else False}
                preds.append(prediction)
                golds.append(item["label"])

            """# Save intermediate validation results
            with open(f"val_results_alpha_{alpha:.4f}.json", "w") as f:
                json.dump(val_final_eval, f, indent=4)"""
            # Calculate Accuracy
            acc = accuracy_score(golds, preds)
            print(f"Accuracy: {acc:.4f}")
            results_log[str(alpha)] = acc

            # Update max score per model
            if acc > max_scores_per_model[MODEL_KEY][0]:
                max_scores_per_model[MODEL_KEY] = (acc, lay_idx, alpha)
                print(f"New max accuracy for {MODEL_KEY}: {acc:.4f} at layer {lay_idx} with alpha {alpha:.4f}")
            # Clean up

            # CRITICAL: Remove hook before next alpha!
            hook_handle.remove()

if __name__ == "__main__":
    for key in model_path_dict.keys():
        # ================= Configuration =================
        MODEL_KEY = key  # Change to select different model from model_path_dict
        MODEL_NAME = model_path_dict[MODEL_KEY] # Check if "Instruct" version is better for your prompt
        LAYER_IDX = model_params_dict[key]["layer_idx"]
        K_NEIGHBORS = 32
        TRAIN_FILE = "train_parsed.json" # Your UUID JSON file
        VAL_FILE = "val_parsed.json"     # Your UUID JSON file
        TEST_FILE = "test_parsed.json"   # Your UUID JSON file
        # =================================================
        print(f"\n\n=== Running Experiment for Model: {MODEL_KEY}, Layer: {LAYER_IDX} ===")
        # Run main experiment
        main(MODEL_NAME=MODEL_NAME, LAYER_IDX=LAYER_IDX, K_NEIGHBORS=K_NEIGHBORS)
        print(f"=== Finished Experiment for Model: {MODEL_KEY}, Layer: {LAYER_IDX} ===\n\n")
    
        # Save results
    with open("results_hyperparams_kcast.json", "w") as f:
        json.dump(max_scores_per_model, f, indent=4)
    print("\nDone. Results saved to results_kcast.json")
