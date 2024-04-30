import numpy as np
import matplotlib.pyplot as plt

cmap = plt.get_cmap('tab10')
plt.set_cmap(cmap)
p = np.load('/home/weimh/code/HW5/output/plot/PatchTST_ETTh1_96/performance.npy')
t = np.load('/home/weimh/code/HW5/output/plot/Transformer_ETTh1_96/performance.npy')
d = np.load('/home/weimh/code/HW5/output/plot/Distillation_ETTh1_96_0.1/performance.npy')

x = np.arange(100) + 1
p_train_loss = p[:, 1]
p_val_loss = p[:, 2]

t_train_loss = t[:, 1]
t_val_loss = t[:, 2]

d_train_loss = 10 * d[:, 1]
d_val_loss = 10 * d[:, 2]

# fig = plt.figure(figsize=(6,5))
# plt.plot(x, p_train_loss, label='p', linewidth=1.5, color=cmap.colors[2])
# plt.plot(x, t_train_loss, label='t', linewidth=1.5, color=cmap.colors[3])
# plt.plot(x, d_train_loss, label='d', linewidth=1.5, color=cmap.colors[4])
# plt.xlabel('Epochs', fontsize=16)
# plt.ylabel('Loss', fontsize=16)
# plt.legend(fontsize=10, loc='lower right')
# plt.grid(ls='--')
# # plt.xlim([0, 101])
# # plt.ylim([0.3, 1.0])
# plt.title('Train Loss')
# plt.savefig('plt loss.png')

fig = plt.figure(figsize=(6,5))
plt.plot(x, p_val_loss, label='PatchTST', linewidth=2, color=cmap.colors[2])
plt.plot(x, t_val_loss, label='Transformer', linewidth=2, color=cmap.colors[3])
plt.plot(x, d_val_loss, label='Distillation', linewidth=2, color=cmap.colors[4])
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Validation Loss', fontsize=16)
plt.legend(fontsize=10, loc='upper right')
plt.grid(ls='--')
# plt.xlim([0, 101])
# plt.ylim([0.3, 1.0])
plt.title('Validation Loss on ETTh1', fontsize=16)
plt.savefig('plt loss.png')
