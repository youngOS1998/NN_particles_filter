import numpy as np
import matplotlib.pyplot as plt

flag_train = True

if flag_train:
    path_name = './loss_save_2.npy'
else:
    path_name = './loss_eval_save.npy'

data = np.load(path_name)

plt.plot(data)
plt.show()
