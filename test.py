import torch

a = torch.tensor([1.3, 2.5, 3.7], requires_grad=True)
b = torch.tensor([4.1, 5.3, 6.5], requires_grad=True)

c = []
c.append(a + b)
c.append(a - b)
c.detach()

print(c)

