import torch
from activations import ActivationExtractor
from dataset_json import load_json_dataset
from kcast import KCASTDatastore, KCASTSteerer
from evaluation import evaluate_model


# ==========================================================
# Load model
# ==========================================================

MODEL = "meta-llama/Llama-3.2-3B"
LAYER = -5
ALPHA = 2.0
K = 16

print(f"Using model: {MODEL}, layer: {LAYER}")
print(f"Steering alpha: {ALPHA}, k: {K}")


extractor = ActivationExtractor(MODEL, layer_idx=LAYER)

train_data = load_json_dataset("train.json")
val_data = load_json_dataset("val.json")
test_data = load_json_dataset("test.json")


# ==========================================================
# Build datastore
# ==========================================================

datastore = KCASTDatastore()
print("Extracting training activations...")

for item in train_data:
    phi = extractor.phi(item["text"])
    datastore.add(phi, item["label"])

datastore.finalize()
print(f"Datastore finalized: {len(datastore.phis)} entries.")


# ==========================================================
# Compute Δφ_c = μ_valid - μ_invalid
# ==========================================================

print("Computing Δφ_c...")

valid_phis = [extractor.phi(d["text"]) for d in train_data if d["label"] == "valid"]
invalid_phis = [extractor.phi(d["text"]) for d in train_data if d["label"] == "invalid"]

mu_valid = torch.stack(valid_phis).mean(0)
mu_invalid = torch.stack(invalid_phis).mean(0)

delta_c = (mu_valid - mu_invalid)
torch.save(delta_c, "delta_vector.pt")
print("Saved Δφ_c → delta_vector.pt")


# ==========================================================
# K-CAST inference (logit-based classifier)
# ==========================================================

def run_kcast(prompt, alpha=ALPHA, k=K):
    layer = extractor.model.model.layers[extractor.layer_idx]

    # register steering hook
    hook = layer.register_forward_hook(
        KCASTSteerer(delta_base=delta_c, datastore=datastore, alpha=alpha, k=k)
    )

    inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.device)

    with torch.no_grad():
        out = extractor.model(**inputs)

    hook.remove()

    logits = out.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)

    tok = extractor.tokenizer
    valid_id = tok.encode("valid", add_special_tokens=False)[0]
    invalid_id = tok.encode("invalid", add_special_tokens=False)[0]

    p_valid = probs[0, valid_id].item()
    p_invalid = probs[0, invalid_id].item()

    return "valid" if p_valid >= p_invalid else "invalid"


# ==========================================================
# Evaluate
# ==========================================================

print("\nEvaluating on validation set...")
val_results = evaluate_model(val_data, run_kcast)
print(val_results)

print("\nEvaluating on TEST set...")
test_results = evaluate_model(test_data, run_kcast)
print(test_results)
