import torch
import torch.nn as nn
from tqdm import tqdm

class MeanSteerer:
    def __init__(self, delta_vector, alpha=1.0):
        """
        delta_vector: The (Mean_Valid - Mean_Invalid) vector.
        alpha: Constant scaling factor for steering intensity.
        """
        # Normalize the delta vector so alpha is consistent across layers
        self.delta = delta_vector / (torch.norm(delta_vector) + 1e-6)
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        # Handle cases where layer output is a tuple (hidden_states, past_key_values)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Apply steering ONLY to the last token (the classification token)
        # We add (alpha * delta) to the existing hidden state
        new_hidden = hidden_states.clone()
        
        # Match device and dtype of the model's activations
        steering_update = self.alpha * self.delta.to(hidden_states.device).to(hidden_states.dtype)
        
        # new_hidden shape: [batch, sequence_length, d_model]
        new_hidden[:, -1, :] += steering_update

        return (new_hidden,) if isinstance(output, tuple) else new_hidden

def get_mean_diff_vector(extractor, train_data, n_samples=500):
    """
    Computes the Mean(Valid) - Mean(Invalid) activation vector.
    """
    valid_vecs = []
    invalid_vecs = []
    
    print(f"Extracting activations for {n_samples} samples...")
    for item in tqdm(train_data[:n_samples]):
        phi = extractor.get_phi(item["text"]) # This gets the last token activation
        if item["label"] == "valid":
            valid_vecs.append(phi)
        else:
            invalid_vecs.append(phi)
            
    if not valid_vecs or not invalid_vecs:
        raise ValueError("Dataset must contain both 'valid' and 'invalid' labels.")

    mean_valid = torch.stack(valid_vecs).mean(dim=0)
    mean_invalid = torch.stack(invalid_vecs).mean(dim=0)
    
    return mean_valid - mean_invalid