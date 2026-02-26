import os
import torch
import json
import numpy as np
from tqdm import tqdm
from dataset_utils import parse_uuid_dataset
from activations import ActivationExtractor
from riser import RiserRouter, RiserSteerer, train_riser

# --- CONFIGURATION ---
MODEL_PATH = "/home/nlp-shared/akash_models/fine-tuned_models/syllogistic_reasoning/model-weights-qlora-syllogisms/Qwen2.5_7B/merged"
TEST_DATA_PATH = "test_data_subtask_1.json"
OUTPUT_FILE = "predictions_riser.json"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# Optimization Parameters
LAYER_IDX = -14
ALPHA = 10.0  # Optimal alpha found from your sweep
MAX_NEW_TOKENS = 256
SYSTEM_MSG = "Take a deep breath and go step by step. Reason through the logic then output 'valid' or 'invalid'."

def generate_with_riser(extractor, input_ids, steerer, max_new_tokens=256):
    """Generates text while applying the RISER steering at every token."""
    generated = input_ids
    
    # Ensure steerer is in generation mode if necessary 
    # (RiserSteerer usually applies to the last token automatically)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = extractor.model(generated)
            logits = outputs.logits[:, -1, :]
            
            # Greedy selection
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            
            if next_token.item() == extractor.tokenizer.eos_token_id:
                break
                
    return extractor.tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    # 1. Initialize Extractor
    extractor = ActivationExtractor(MODEL_PATH, layer_idx=LAYER_IDX)
    
    # 2. Combine Data for Training the Steering Vector and Router
    print("Loading and combining datasets...")
    train_data = parse_uuid_dataset("train_parsed.json")
    val_data = parse_uuid_dataset("val_parsed.json")
    combined_data = train_data + val_data
    
    # 3. Compute Delta Vector
    print("Computing Delta Vector (Valid - Invalid)...")
    valid_vecs = [extractor.get_phi(i["text"]) for i in combined_data[:400] if i["label"] == "valid"]
    invalid_vecs = [extractor.get_phi(i["text"]) for i in combined_data[:400] if i["label"] == "invalid"]
    delta_c = torch.stack(valid_vecs).mean(0) - torch.stack(invalid_vecs).mean(0)
    
    # 4. Train the Dynamic RISER Router
    print("Training RISER Router...")
    # Using a larger slice for better routing precision
    router = train_riser(extractor, delta_c, combined_data[:800])
    
    # 5. Setup Steerer and Hook
    steerer = RiserSteerer(delta_c, router, global_alpha=ALPHA)
    target_layer = extractor.model.model.layers[LAYER_IDX]
    hook_handle = target_layer.register_forward_hook(steerer)

    # 6. Run Inference on Test Data
    with open(TEST_DATA_PATH, "r") as f:
        test_items = json.load(f)

    final_results = []
    print(f"Generating RISER-steered responses for {len(test_items)} items...")

    for item in tqdm(test_items):
        # Apply the specific prompting template
        prompt = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{item['syllogism']}\nDecision:<|im_end|>\n<|im_start|>assistant\n"
        
        input_ids = extractor.tokenizer(prompt, return_tensors="pt").input_ids.to(extractor.device)
        
        # Generate full CoT response
        full_text = generate_with_riser(extractor, input_ids, steerer, MAX_NEW_TOKENS)
        
        # Parse decision (looking after the "assistant" tag/decision prompt)
        response_part = full_text.split("Decision:")[-1].lower()
        
        if "invalid" in response_part:
            prediction = False
        elif "valid" in response_part:
            prediction = True
        else:
            # Fallback logic: check whole text if 'Decision:' split failed
            prediction = "invalid" not in full_text.lower() and "valid" in full_text.lower()

        final_results.append({
            "id": item["id"],
            "validity": prediction
        })

    # 7. Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_results, f, indent=4)

    # Cleanup
    hook_handle.remove()
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()