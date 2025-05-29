import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as sio

disp_x = sio.loadmat('../DIC-dataset/gt3/train_image_1.mat')['Disp_field_1'][0].astype(float)

fig, ax = plt.subplots(figsize=(disp_x.shape[1]/200, disp_x.shape[0]/200))
c = plt.pcolormesh(disp_x,vmin = -0.5, vmax = 0.5)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(c, cax=cax)
plt.show()