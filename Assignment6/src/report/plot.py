import sys
import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

kernels = []
data = []
if len(sys.argv) != 2:
    print("Execute as :\npython {} {}".format(sys.argv[0],"filename.txt"))
    exit(1)
with open(sys.argv[1]) as f:
    for lineNum,line in enumerate(f):
        if 0 == lineNum:
            kernels = eval(line.strip())
        else: 
            kernelNum, vals = line.strip().split(":")
            data.append(eval(vals))

data = np.asarray(data).T
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
plt.xticks(range(14))
plt.legend(loc='lower right')
plt.xlabel("13 functional categories")
plt.ylabel("Accuracy (in percent)")
plt.grid()
plt.show()

# func_props = data.shape[1]
# kernel_count = data.shape[0]
# 
# cNorm  = colors.Normalize(vmin=0, vmax=len(data.T))
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
# f, axarr = plt.subplots(1, func_props)
# for idx, data_func_prop in enumerate(data.T):
#     arr = axarr[idx]
#     # colorVal = scalarMap.to_rgba(idx)
#     # handle = arr.scatter([0]*kernel_count,data_func_prop*100,color=colorVal,label=kernel)
#     colorVal = scalarMap.to_rgba(idx)
#     handle = arr.scatter([0]*kernel_count,data_func_prop*100)
# 
#     arr.set_title("Prop = {}".format(idx))
#     arr.grid()
#     arr.tick_params(axis='x',which='both',bottom='off',labelbottom='off')
# 
# arr.legend(loc='center right')
# # plt.xlabel("13 functional categories")
# # plt.ylabel("Accuracy (in percent)")
# # plt.grid()
# plt.show()



