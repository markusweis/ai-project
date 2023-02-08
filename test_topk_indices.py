import torch
# N = 5
x = torch.rand(15, 200)
y = torch.rand(15, 200)

concated = torch.cat((x,y), 1)

print("Done.")