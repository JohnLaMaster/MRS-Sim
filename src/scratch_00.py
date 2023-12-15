# %%
import numpy as np
import torch
# %%
mn, mx = np.sqrt(20), np.sqrt(100)
# %%
print(mn, mx)
print(type(mn))
# %%
test = torch.randint(low=20, high=100, size=(1,))#.sqrt()
print(test)
print(test.squeeze())
# %%
p = []
for _ in range(10):
    p.append(torch.bernoulli(torch.tensor(0.8)))
print(p)
# %%
