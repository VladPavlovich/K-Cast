
import torch
import torch.nn.functional as F

class KCASTDatastore:
    def __init__(self):
        # Each entry = (activation_phi, label)
        self.store = []

    def add(self, phi, label):
        self.store.append((phi.cpu(), label))

    def knn_label(self, phi, k=15):
        distances = []
        for vec, label in self.store:
            d = torch.norm(phi - vec.to(phi.device)).item()
            distances.append((d, label))

        distances.sort(key=lambda x: x[0])
        topk = distances[:k]
        labels = [lbl for _, lbl in topk]

        # return majority vote
        return max(set(labels), key=labels.count)


### ---- Steering hook ----

class KCASTSteerer:
    def __init__(self, delta_vec, datastore, alpha=1.0):
        self.delta = delta_vec
        self.datastore = datastore
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        # output shape: (B, S, H)
        B, S, H = output.shape

        # activation before steering
        phi = output[:, -1, :].squeeze()

        predicted_label = self.datastore.knn_label(phi)

        # determine steering direction
        sign = 1 if predicted_label == "invalid" else -1

        # apply K-CAST update
        new_out = output.clone()
        new_out[:, -1, :] += sign * self.alpha * self.delta.to(output.device)

        return new_out
