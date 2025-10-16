import torch
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Register forward and backward hooks on target layer
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        # Zero grads
        self.model.zero_grad()
        # Backward pass for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute weights: global average pool of gradients
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight activations
        activations = self.activations[0]
        for i in range(len(pooled_grads)):
            activations[i, :, :] *= pooled_grads[i]

        # Compute heatmap
        heatmap = torch.sum(activations, dim=0).cpu()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.numpy()

        return heatmap
