# kcast.py

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------
# K-CAST Datastore: stores (activation, label) pairs for k-NN
# --------------------------------------------------------------------

class KCASTDatastore:
    def __init__(self):
        self.store = []  # list of (phi, label)

    def add(self, phi, label):
        self.store.append((phi.cpu(), label))

    def knn_label(self, phi, k=15):
        """
        Returns majority-vote label among k nearest activations.
        """
        distances = []
        for vec, label in self.store:
            d = torch.norm(phi - vec.to(phi.device)).item()
            distances.append((d, label))

        distances.sort(key=lambda x: x[0])
        topk = distances[:k]

        labels = [lbl for _, lbl in topk]
        return max(set(labels), key=labels.count)


# --------------------------------------------------------------------
# Steering Hook for K-CAST
# This applies: phi' = phi + sign * alpha * delta
# --------------------------------------------------------------------

class KCASTSteerer:
    def __init__(self, delta, datastore, alpha=1.0):
        self.delta = delta
        self.datastore = datastore
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        # Extract last-token activation
        phi = output[:, -1, :].squeeze()

        # Predict label via k-NN
        predicted_label = self.datastore.knn_label(phi)

        # Determine steering direction
        sign = 1 if predicted_label == "invalid" else -1

        # Apply steering update
        new_output = output.clone()
        new_output[:, -1, :] += sign * self.alpha * self.delta.to(output.device)

        return new_output
