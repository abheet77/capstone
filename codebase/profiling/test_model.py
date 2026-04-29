import torch
import torchvision.models as models
import time

device = torch.device("cuda")

model = models.mobilenet_v2(pretrained=True).to(device)
model.eval()

x = torch.randn(1, 3, 224, 224).to(device)

# warmup
for _ in range(10):
    _ = model(x)

torch.cuda.synchronize()
start = time.time()

for _ in range(50):
    _ = model(x)

torch.cuda.synchronize()
end = time.time()

print("Device:", device)
print("Avg inference time:", (end - start)/50 * 1000, "ms")