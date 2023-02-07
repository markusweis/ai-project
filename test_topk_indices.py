import torch
# N = 5
adj = torch.randint(low=0, high=1, size=(5,5))
x = torch.rand(10)

y = torch.topk(x, 4).indices
indieces = torch.triu_indices(5,5, 1)
transposed = indieces.t()

selected_edges = transposed[y]


print("Done.")