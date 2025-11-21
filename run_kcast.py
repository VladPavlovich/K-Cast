import torch
from activations import ActivationExtractor
from dataset_json import load_json_dataset
from kcast import KCASTDatastore, KCASTSteerer
from evaluation import evaluate_model


# -------------------------------------------------------
# Load model + data
# -------------------------------------------------------

MODEL = "meta-llama/Llama-3.2-1B"  # change if you want
extractor = ActivationExtractor(MODEL)

train_data = load_json_dataset("train.json")
val_data = load_json_dataset("val.json")
test_data = load_json_dataset("test.json")


# -------------------------------------------------------
# Build K-CAST datastore using training set
# -------------------------------------------------------

datastore = KCASTDatastore()

print("Extracting training activations for K-CAST datastore...")
for item in train_data:
    phi = extractor.phi(item["text"])
    datastore.add(phi, item["label"])


# -------------------------------------------------------
# Compute steering vector (μ_valid - μ_invalid)
# -------------------------------------------------------

print("Computing delta (steering vector) from training set...")

valid_phis = [extractor.phi(d["text"]) for d in train_data if d["label"] == "valid"]
invalid_phis = [extractor.phi(d["text"]) for d in train_data if d["label"] == "invalid"]

mu_valid = torch.stack(valid_phis).mean(0)
mu_invalid = torch.stack(invalid_phis).mean(0)
delta = mu_valid - mu_invalid

torch.save(delta, "delta_vector.pt")
print("Saved delta_vector.pt")


# -------------------------------------------------------
# Define run_kcast function for evaluation
# -------------------------------------------------------

def run_kcast(prompt, alpha=1.0):
    layer = extractor.model.model.layers[extractor.layer_idx]
    hook = layer.register_forward_hook(KCASTSteerer(delta, datastore, alpha))

    inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.device)
    with torch.no_grad():
        out = extractor.model.generate(**inputs, max_new_tokens=50)

    hook.remove()
    return extractor.tokenizer.decode(out[0], skip_special_tokens=True)


# -------------------------------------------------------
# Evaluate on validation + test sets
# -------------------------------------------------------

print("\nEvaluating on validation set...")
val_results = evaluate_model(val_data, run_kcast)
print(val_results)

print("\nEvaluating on TEST set...")
test_results = evaluate_model(test_data, run_kcast)
print(test_results)
