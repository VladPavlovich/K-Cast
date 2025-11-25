import torch
import torch.nn as nn
import torch.nn.functional as F


class KCASTDatastore:
    """
    Stores activation vectors and labels for kNN-based α computation.
    Paper: K-CAST computes α per example from kNN in activation space.
    Our version: we store for compatibility, but α is fixed unless extended.
    """

    def __init__(self):
        self.vecs = []      # activation vectors
        self.labels = []    # "valid" / "invalid"

    def add(self, vec, label):
        self.vecs.append(vec.cpu())
        self.labels.append(1 if label == "valid" else 0)

    def __len__(self):
        return len(self.vecs)


class KCASTSteerer(nn.Module):
    """
    The forward hook injected into a transformer block.

    It modifies the hidden state h:
        h' = h + α * δ
    where δ is the steering vector.
    """

    def __init__(self, delta, datastore, alpha=1.0):
        super().__init__()
        self.delta = delta.to(dtype=torch.float32)
        self.alpha = alpha
        self.datastore = datastore  # not used yet, but needed for method 2

    def __call__(self, module, inputs, output):
        out = output

        # ensure delta is on same device (important!)
        if self.delta.device != out.device:
            self.delta = self.delta.to(out.device)

        # Apply steering to the LAST token hidden state
        out[:, -1, :] = out[:, -1, :] + self.alpha * self.delta

        return out
