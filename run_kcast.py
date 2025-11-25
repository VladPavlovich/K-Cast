import torch
from activations import ActivationExtractor
from dataset_json import load_json_dataset
from kcast import KCASTDatastore, KCASTSteerer
from evaluation import evaluate_model


# -------------------------------------------------------
# Load model & data
# -------------------------------------------------------

MODEL = "meta-llama/Llama-3.2-3B"  # change if needed
extractor = ActivationExtractor(MODEL)

train_data = load_json_dataset("train.json")
val_data = load_json_dataset("val.json")
test_data = load_json_dataset("test.json")

device = extractor.device
tokenizer = extractor.tokenizer
model = extractor.model


# -------------------------------------------------------
# Get token IDs for "valid" and "invalid"
# -------------------------------------------------------

valid_id = tokenizer("valid", add_special_tokens=False).input_ids[0]
invalid_id = tokenizer("invalid", add_special_tokens=False).input_ids[0]

print("Valid token ID:", valid_id)
print("Invalid token ID:", invalid_id)


# -------------------------------------------------------
# Build K-CAST datastore using training data
# -------------------------------------------------------

datastore = KCASTDatastore()

print("Extracting training activations for datastore...")
for item in train_data:
    phi_vec = extractor.phi(item["text"])
    datastore.add(phi_vec, item["label"])


# -------------------------------------------------------
# Compute delta = μ_valid – μ_invalid
# -------------------------------------------------------

print("Computing delta (steering vector)...")

valid_phis = [extractor.phi(d["text"]) for d in train_data if d["label"] == "valid"]
invalid_phis = [extractor.phi(d["text"]) for d in train_data if d["label"] == "invalid"]

mu_valid = torch.stack(valid_phis).mean(0)
mu_invalid = torch.stack(invalid_phis).mean(0)
delta = mu_valid - mu_invalid

torch.save(delta, "delta_vector.pt")
print("Saved delta_vector.pt")


# -------------------------------------------------------
# Logit-based K-CAST classifier (paper method)
# -------------------------------------------------------

def run_kcast(prompt, alpha=1.0):
    """
    Runs a single classification using K-CAST on logits.
    No generation. No free-form output.
    """
    # Select model layer for intervention
    layer = model.model.layers[extractor.layer_idx]

    # Register K-CAST intervention hook
    hook = layer.register_forward_hook(KCASTSteerer(delta, datastore, alpha))

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs)

    hook.remove()  # clean up hook

    logits = out.logits[:, -1, :]     # final-token logits

    # Compare logits for valid vs invalid
    valid_logit = logits[0, valid_id].item()
    invalid_logit = logits[0, invalid_id].item()

    pred = "valid" if valid_logit > invalid_logit else "invalid"

    return pred


# -------------------------------------------------------
# Evaluate on validation & test data
# -------------------------------------------------------

print("\nEvaluating on validation set...")
val_results = evaluate_model(val_data, run_kcast)
print(val_results)

print("\nEvaluating on TEST set...")
test_results = evaluate_model(test_data, run_kcast)
print(test_results)
