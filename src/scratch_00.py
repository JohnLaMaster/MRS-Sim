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
def enforce_divisibility(input, divisor):
    if not 1 / (input % divisor) == 0:
        rem = 1 / (input % divisor) / 10
        if rem >= 0.5:
            input += (1 - rem) * divisor
        else:
            input -= rem * divisor
    return int(input)        

# %%
print(enforce_divisibility(8,68))
# %%
a = 100
a //= 10
print(a, type(a))
# %%
