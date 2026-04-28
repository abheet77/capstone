import torch
import torchvision.models as models
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=True).to(device)
model.eval()

x = torch.randn(1, 3, 224, 224).to(device)

# warmup
for _ in range(10):
    _ = model(x)

# timing
start = time.time()
for _ in range(50):
    _ = model(x)
end = time.time()

avg_ms = (end - start) / 50 * 1000
print(f"Device: {device}")
print(f"Avg inference time: {avg_ms:.2f} ms")