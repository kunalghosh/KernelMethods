import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

kernels = []
data = []
with open("plot_data_orig.txt") as f:
    for line in f:
        kernel, vals = line.strip().split(":")
        kernels.append(kernel.strip())
        data.append(eval(vals))

data = np.asarray(data)
jet = cm = plt.get_cmap('Dark2') 
cNorm  = colors.Normalize(vmin=0, vmax=len(kernels))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

x_axis = range(1,len(data[0])+1)
# handles = []
for idx, kernel in enumerate(kernels):
    colorVal = scalarMap.to_rgba(idx)
    handle = plt.scatter(x_axis,data[idx,:]*100,color=colorVal,label=kernel)
    plt.plot(x_axis,data[idx,:]*100,color=colorVal)
    # print(handle.get_label())
    # print(handle)
    # handles.append(handle)

# plt.legend(handles,loc='lower right')
plt.legend(loc='lower right')
plt.xlabel("13 functional categories")
plt.ylabel("Accuracy (in percent)")
plt.grid()
plt.show()
