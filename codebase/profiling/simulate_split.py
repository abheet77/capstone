import torch
import torchvision.models as models
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=True)
model.eval()

# 🔴 split at layer 7 (based on your data)
part1 = torch.nn.Sequential(*model.features[:7]).to(device)
part2 = torch.nn.Sequential(*model.features[7:]).to("cpu")

x = torch.randn(1, 3, 224, 224).to(device)

start = time.time()

# GPU part
x = part1(x)

# simulate transfer
tensor_size = x.element_size() * x.nelement() / (1024 * 1024)
bandwidth = 100  # MB/s

delay = tensor_size / bandwidth
time.sleep(delay)

x = x.cpu()

# CPU part (mock FPGA)
x = part2(x)

end = time.time()

print("Tensor size:", tensor_size, "MB")
print("Simulated delay:", delay, "sec")
print("Total execution time:", (end - start) * 1000, "ms")