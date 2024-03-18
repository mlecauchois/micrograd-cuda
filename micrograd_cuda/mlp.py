import random
import json
import os

from micrograd_cuda.tensor import Tensor


class Layer:

    def __init__(self, nin, nout, label=None):
        label = label if label is not None else f"nin:{nin} nout:{nout}"
        self.w = Tensor(
            [[random.uniform(-1, 1) for _ in range(nin + 1)] for _ in range(nout)],
            label=label,
        )

    def __call__(self, x):
        bias_input = Tensor([[1.0 for _ in range(x.shape[1])]], requires_grad=False)
        bias_input.to(x.device)
        x_with_bias = x.concat(bias_input, axis=0)
        outs = self.w @ x_with_bias
        outs = outs.tanh()
        return outs

    def parameters(self):
        return [self.w]
    
    def to(self, device: str):
        for parameter in self.parameters():
            parameter.to(device)


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def state_dict(self):
        state = {}
        for layer in self.layers:
            state[layer.w.label] = layer.w.data
        return state

    def save(self, directory_path):
        # Ensure the directory exists
        os.makedirs(directory_path, exist_ok=True)

        # Save the weights
        weights_path = os.path.join(directory_path, "weights.json")
        with open(weights_path, "w") as f:
            json.dump(self.state_dict(), f, indent=4)

        # Save the configuration
        config_path = os.path.join(directory_path, "config.json")
        architecture = [layer.w.shape[1] - 1 for layer in self.layers] + [
            self.layers[-1].w.shape[0]
        ]  # Extract layer sizes
        with open(config_path, "w") as f:
            json.dump(architecture, f, indent=4)

    @classmethod
    def load(cls, directory_path):
        # Load the configuration
        with open(os.path.join(directory_path, "config.json"), "r") as f:
            config = json.load(f)

        # Initialize the model with the architecture from the config
        model = cls(config[0], config[1:])

        # Load the weights
        with open(os.path.join(directory_path, "weights.json"), "r") as f:
            weights = json.load(f)

        # Assign the weights to the model's layers
        for layer in model.layers:
            layer.w.data = weights[layer.w.label]

        return model
    
    def to(self, device: str):
        for parameter in self.parameters():
            parameter.to(device)
