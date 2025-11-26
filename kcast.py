import torch
import torch.nn.functional as F

class KCASTDatastore:
    """
    Stores (phi(x), label) pairs and performs kNN queries using Cosine Similarity.
    Labels: valid (+1), invalid (-1).
    """
    def __init__(self, device="cuda"):
        self.phis = []
        self.labels = []
        self.device = device

    def add(self, phi, label):
        # phi is expected to be [hidden_dim]
        # We ensure it's on the correct device immediately
        self.phis.append(phi.to(self.device))
        self.labels.append(1 if label == "valid" else -1)

    def finalize(self):
        if not self.phis:
            print("Warning: Datastore is empty!")
            return
            
        # Stack into a single tensor [N, d]
        self.phis_tensor = torch.stack(self.phis)
        
        # Pre-normalize for fast Cosine Similarity: A . B / (|A|*|B|)
        self.phis_norm = F.normalize(self.phis_tensor, p=2, dim=1)
        self.labels_tensor = torch.tensor(self.labels, device=self.device)

    def query_knn(self, phi_x, k=32):
        """
        Returns majority-vote label (+1 or -1) using Cosine Similarity.
        """
        # Ensure input is on the right device
        phi_x = phi_x.to(self.device)
        
        # Normalize query vector: [1, d]
        phi_x_norm = F.normalize(phi_x.unsqueeze(0), p=2, dim=1)
        
        # Dot product with all stored vectors: [1, N]
        sims = torch.mm(phi_x_norm, self.phis_norm.t()).squeeze()
        
        # Get top-k indices (largest similarity)
        # Ensure k doesn't exceed datastore size
        k = min(k, len(self.labels))
        topk_indices = torch.topk(sims, k).indices
        
        # Get labels of neighbors
        neighbor_labels = self.labels_tensor[topk_indices]
        
        # Majority vote: if sum >= 0, then valid, else invalid
        return 1 if neighbor_labels.sum() >= 0 else -1


class KCASTSteerer:
    """
    Applies steering to the forward pass.
    Formula: phi_new = phi - y_hat * alpha * delta
    """
    def __init__(self, delta_base, datastore, alpha=1.0, k=32):
        self.delta_base = delta_base
        self.datastore = datastore
        self.alpha = alpha
        self.k = k

    def __call__(self, module, inputs, output):
        # Handle HF output tuple (hidden_states, past_key_values, attentions)
        if isinstance(output, tuple):
            hidden_states = output[0]
            is_tuple = True
        else:
            hidden_states = output
            is_tuple = False

        # Extract last token vector for the query [Batch, Seq, Dim] -> [Dim]
        # We assume batch_size=1 for this implementation.
        # If batch > 1, this needs a loop or vectorized logic.
        last_token_vec = hidden_states[0, -1, :]
        
        # Detach for kNN query (we don't want to backprop through the query)
        query_vec = last_token_vec.detach()

        # 1. Predict label via kNN (y_hat)
        y_hat = self.datastore.query_knn(query_vec, k=self.k)

        # 2. Prepare Delta
        delta = self.delta_base.to(hidden_states.device)
        
        # 3. Apply Update (Cloning to ensure we don't modify in-place errors)
        new_hidden = hidden_states.clone()
        
        # Apply steering to the last token
        update = y_hat * self.alpha * delta
        new_hidden[:, -1, :] = hidden_states[:, -1, :] - update

        # Return tuple if input was tuple
        if is_tuple:
            return (new_hidden,) + output[1:]
        return new_hidden