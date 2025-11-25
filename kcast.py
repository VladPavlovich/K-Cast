import torch

class KCASTDatastore:
    """
    Stores (phi(x), label) pairs and performs kNN queries.
    Labels are: valid → +1, invalid → -1
    """

    def __init__(self):
        self.phis = []
        self.labels = []

    def add(self, phi, label):
        self.phis.append(phi)
        self.labels.append(1 if label == "valid" else -1)

    def finalize(self):
        self.phis = torch.stack(self.phis)          # [N, d]
        self.labels = torch.tensor(self.labels)     # [N]

    def query_knn(self, phi_x, k=32):
        """
        Returns majority-vote label (+1 or -1)
        """
        dists = torch.norm(self.phis - phi_x, dim=1)
        idx = torch.topk(-dists, k).indices        # nearest = largest similarity

        neighbor_labels = self.labels[idx]
        return 1 if neighbor_labels.sum() >= 0 else -1


class KCASTSteerer:
    """
    Implements:
      φ̃(x) = φ(x) - ŷ(x) * α * Δφ_c
    where ŷ(x) is obtained via kNN majority vote.
    """

    def __init__(self, delta_base, datastore, alpha=1.0, k=32):
        self.delta_base = delta_base
        self.datastore = datastore
        self.alpha = alpha
        self.k = k

    def __call__(self, module, inputs, output):
        """
        Apply steering to *last token* activation.
        Output is hidden states: [batch, seq, dim]
        """
        hidden = output
        last_vec = hidden[:, -1, :].detach().cpu().squeeze()

        # 1) Predict label via kNN
        y_hat = self.datastore.query_knn(last_vec, k=self.k)   # +1 or -1

        # 2) Steering direction
        delta = self.delta_base.to(hidden.device)

        # 3) Apply update
        new_hidden = hidden.clone()
        new_hidden[:, -1, :] = hidden[:, -1, :] - y_hat * self.alpha * delta

        return new_hidden
