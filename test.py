import torch
torch.cuda.empty_cache()
print(torch.cuda.device_count())