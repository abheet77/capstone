import torch
import torchvision.models as models
import time

device = torch.device("cuda")

model = models.mobilenet_v2(pretrained=True).to(device)
model.eval()

x = torch.randn(1, 3, 224, 224).to(device)

print("\nLayer-wise latency:\n")

for i, layer in enumerate(model.features):
    torch.cuda.synchronize()
    start = time.time()

    x = layer(x)

    torch.cuda.synchronize()
    end = time.time()

    print(f"Layer {i}: {(end - start)*1000:.3f} ms")