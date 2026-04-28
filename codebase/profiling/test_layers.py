import torch
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=True).to(device)
model.eval()

x = torch.randn(1, 3, 224, 224).to(device)

print("\nLayer-wise tensor sizes:\n")

for i, layer in enumerate(model.features):
    x = layer(x)
    size_mb = x.element_size() * x.nelement() / (1024 * 1024)
    print(f"Layer {i}: shape={x.shape}, size={size_mb:.2f} MB")