import torch
import torchvision.models as models

# ---- timing function (required) ----
def time_model(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    for _ in range(iters):
        fn()
    ender.record()

    torch.cuda.synchronize()
    return starter.elapsed_time(ender) / iters
# -----------------------------------

device = torch.device("cuda")

model = models.mobilenet_v2(pretrained=True).to(device)
model.eval()

# 🔴 change this in lab: 5 / 7 / 10
SPLIT_INDEX = 7

part1 = torch.nn.Sequential(*model.features[:SPLIT_INDEX]).to(device)
part2 = torch.nn.Sequential(*model.features[SPLIT_INDEX:]).to(device)

x = torch.randn(1, 3, 224, 224).to(device)

def run():
    y = part1(x)
    y = part2(y)

ms = time_model(run, iters=100, warmup=20)

print(f"Split@{SPLIT_INDEX} (GPU→GPU): {ms:.3f} ms")