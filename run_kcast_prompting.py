import torch
import random
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# Import our modules
from dataset_utils import parse_uuid_dataset
from activations import ActivationExtractor
from kcast import KCASTDatastore, KCASTSteerer



# ================= Configuration =================
MODEL_NAME = "/home/nlp-shared/akash_models/fine-tuned_models/syllogistic_reasoning/model-weights-qlora-syllogisms/Qwen2.5_7B/merged"
LAYER_IDX = [-14]
K_NEIGHBORS = 32
TRAIN_FILE = "train_parsed.json" # Your UUID JSON file
VAL_FILE = "val_parsed.json"     # Your UUID JSON file
# =================================================

def main():
    # Load model and data once
    extractor = ActivationExtractor(MODEL_NAME, layer_idx=LAYER_IDX[0])
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


        for i, item in enumerate(tqdm(train_data)):
            prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{item['text']}\nDecision:<|im_end|>\n<|im_start|>assistant\n"
            valid_vecs = [extractor.get_phi(i["text"]) for i in train_data if i["label"] == "valid"]
            invalid_vecs = [extractor.get_phi(i["text"]) for i in train_data if i["label"] == "invalid"]
            
            phi = extractor.get_phi(prompt)
            
            if item["label"] == "valid":
                valid_vecs.append(phi)
            else:
                invalid_vecs.append(phi)
            
            if i % 100 == 0: print(f"  Processed {i}/{len(train_data)}")

        #compute delta
        mu_valid = torch.stack(valid_vecs).mean(dim=0)
        mu_invalid = torch.stack(invalid_vecs).mean(dim=0)
        delta_c = mu_valid - mu_invalid


        alpha_list = [0.0]
        results_table = []

        for alpha in alpha_list:
            print(f"\n{'-'*10} ALPHA: {alpha} {'-'*10}")
            
            steerer = KCASTSteerer(
                datastore,
                delta_c,
                k=K_NEIGHBORS,
                alpha=alpha,
                layer_idx=lay_idx,
                device=extractor.device
            )
            steerer.gamma = GAMMA
            
            # evaluate kcast steered
            preds, golds = [], []
            for item in tqdm(val_data):
                prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
                prompt += f"<|im_start|>user\n{item['text']}\nDecision:<|im_end|>\n<|im_start|>assistant\n"
                
                inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.device)
                logits = get_dcm_logits(extractor, inputs, steerer)
                
                v_score = logits[valid_id].item()
                inv_score = logits[invalid_id].item()
                
                preds.append("valid" if v_score > inv_score else "invalid")
                golds.append(item["label"])
            
            acc = accuracy_score(golds, preds)
            print(f"Steered Accuracy: {acc:.4f}")
            results_table.append((lay_idx, alpha, acc))
    print("\nFinal Results:")
    for row in results_table:
        print(f"Layer: {row[0]} | Alpha: {row[1]} | Accuracy: {row[2]:.4f}")
        preds.append("valid" if v_score > inv_score else "invalid")
        golds.append(item["label"])
    base_acc = accuracy_score(golds, preds)
    print(f"Baseline Accuracy: {base_acc:.4f}")         


if __name__ == "__main__":
    main()  

    



        