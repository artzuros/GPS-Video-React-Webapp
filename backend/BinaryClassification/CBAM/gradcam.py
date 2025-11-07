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
            # Keep activations on the device
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            # Keep gradients on the device
            self.gradients = grad_out[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: torch.Tensor, shape [1, C, H, W] on correct device
        class_idx: optional int, target class index
        """
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Zero grads and backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output, device=output.device)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # --- Vectorized Grad-CAM computation on GPU ---
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])  # [C]
        # Weighted combination of activations (broadcasted multiplication)
        activations = self.activations[0] * pooled_grads[:, None, None]  # [C, H, W]

        # Compute heatmap
        heatmap = torch.relu(torch.sum(activations, dim=0))  # [H, W]
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.cpu().numpy()  # convert to NumPy for OpenCV

        return heatmap
