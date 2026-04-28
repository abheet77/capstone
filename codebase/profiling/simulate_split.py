import torch
import torchvision.models as models
import time

device = torch.device("cuda")

model = models.mobilenet_v2(pretrained=True).to(device)
model.eval()

# split at layer 7
part1 = torch.nn.Sequential(*model.features[:7]).to(device)
part2 = torch.nn.Sequential(*model.features[7:]).to(device)

x = torch.randn(1, 3, 224, 224).to(device)

# warmup
for _ in range(10):
    _ = part2(part1(x))

start = time.time()

for _ in range(50):
    out = part1(x)
    out = part2(out)

end = time.time()

print("Split GPU→GPU time:", (end - start)/50 * 1000, "ms")