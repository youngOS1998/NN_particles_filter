import numpy as np
import matplotlib.pyplot as plt

data = np.load('./loss_save.npy')

plt.plot(data)
plt.show()
