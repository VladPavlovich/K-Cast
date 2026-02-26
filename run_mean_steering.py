import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset_utils import parse_uuid_dataset
from activations import ActivationExtractor
from mean_steering import MeanSteerer, get_mean_diff_vector

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

MODEL_PATH = "/home/nlp-shared/akash_models/fine-tuned_models/syllogistic_reasoning/model-weights-qlora-syllogisms/Llama_3.2_3B/merged"
# Testing the late layers you found promising
LAYERS_TO_TEST = [-14]
ALPHA_VALUES = [1.0, 1.5,2.0,3.0,5.0,6.0 ,10.0, 20.0,25.0,30.0 ]

def evaluate(extractor, val_data, valid_id, invalid_id):
    preds, golds = [], []
    for item in val_data:
        inputs = extractor.tokenizer(item["text"], return_tensors="pt").to(extractor.device)
        with torch.no_grad():
            out = extractor.model(**inputs)
        
        logits = out.logits[0, -1, :]
        v_score = logits[valid_id].item()
        inv_score = logits[invalid_id].item()
        
        preds.append("valid" if v_score > inv_score else "invalid")
        golds.append(item["label"])
    return accuracy_score(golds, preds)

def main():
    train_data = parse_uuid_dataset("train_parsed.json")
    val_data = parse_uuid_dataset("val_parsed.json")
    
    for lay_idx in LAYERS_TO_TEST:
        print(f"\n" + "="*50)
        print(f"LAYER: {lay_idx}")
        print("="*50)
        
        extractor = ActivationExtractor(MODEL_PATH, layer_idx=lay_idx)
        valid_id = extractor.tokenizer.encode("valid", add_special_tokens=False)[0]
        invalid_id = extractor.tokenizer.encode("invalid", add_special_tokens=False)[0]
        
        # 1. Baseline Accuracy (No Steering)
        base_acc = evaluate(extractor, val_data, valid_id, invalid_id)
        print(f"Baseline Accuracy: {base_acc:.4f}")
        
        # 2. Compute Mean Difference Vector
        delta_vector = get_mean_diff_vector(extractor, train_data)
        
        # 3. Alpha Sweep
        target_layer = extractor.model.model.layers[lay_idx]
        
        for alpha in ALPHA_VALUES:
            steerer = MeanSteerer(delta_vector, alpha=alpha)
            hook = target_layer.register_forward_hook(steerer)
            
            steered_acc = evaluate(extractor, val_data, valid_id, invalid_id)
            print(f"Alpha: {alpha:<4} | Accuracy: {steered_acc:.4f} | Change: {steered_acc - base_acc:+.4f}")
            
            hook.remove()
            torch.cuda.empty_cache()

        del extractor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()