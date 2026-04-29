import torch
import torchvision.models as models

# ---- timing function (YOU MISSED THIS) ----
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
# ------------------------------------------

device = torch.device("cuda")

model = models.mobilenet_v2(pretrained=True).to(device)
model.eval()

x = torch.randn(1, 3, 224, 224).to(device)

def run():
    _ = model(x)

ms = time_model(run, iters=100, warmup=20)

print("Baseline (full GPU):", ms, "ms")