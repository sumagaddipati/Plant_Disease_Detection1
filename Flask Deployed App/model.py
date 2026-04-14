import torch.nn.functional as F

probs = F.softmax(torch.tensor(output), dim=0)
print(probs)