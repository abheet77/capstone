import torch
import torchvision.models as models
import time

device = torch.device("cuda")

model = models.mobilenet_v2(pretrained=True).to(device)
model.eval()

# 🔴 CHANGE THIS VALUE IN LAB: 5 / 7 / 10
SPLIT_INDEX = 7

part1 = torch.nn.Sequential(*model.features[:SPLIT_INDEX]).to(device)
part2 = torch.nn.Sequential(*model.features[SPLIT_INDEX:]).to(device)

x = torch.randn(1, 3, 224, 224).to(device)

# warmup
for _ in range(10):
    _ = part2(part1(x))

torch.cuda.synchronize()
start = time.time()

for _ in range(50):
    out = part1(x)
    out = part2(out)

torch.cuda.synchronize()
end = time.time()

print(f"Split index: {SPLIT_INDEX}")
print("Split GPU→GPU time:", (end - start)/50 * 1000, "ms")