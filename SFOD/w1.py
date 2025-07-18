import torch
from matplotlib import pyplot as plt
plt.style.use('science')
fig = plt.figure(dpi=200)
x = torch.arange(-5, 5, 0.0001)

y = torch.round(torch.log2(x.abs()))

z = torch.sign(x) * torch.pow(2, y)

plt.plot(x, z, label='$\\mathcal{Q}(x)$')
plt.xlabel('$\\boldsymbol{W}$')
plt.ylabel('$\\boldsymbol{W}_{q}$')
plt.legend()
plt.savefig('./pow2_quantizer.pdf')
