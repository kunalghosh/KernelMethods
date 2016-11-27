import numpy as np
from sklearn import svm
from get_alignment import get_alignment

if __name__ == '__main__':

    # Load data

    kernel_files = ['~/Workspace/datasets/KernelMethods/Assignment6/data/K_exp.txt',
                    '~/Workspace/datasets/KernelMethods/Assignment6/data/K_mgi.txt',
                    '~/Workspace/datasets/KernelMethods/Assignment6/data/K_mpi.txt',
                    '~/Workspace/datasets/KernelMethods/Assignment6/data/K_pfamdom.txt',
                    '~/Workspace/datasets/KernelMethods/Assignment6/data/K_sw.txt',
                    '~/Workspace/datasets/KernelMethods/Assignment6/data/K_tap.txt']

    Kernels    = [np.loadtxt(file) for file in kernel_files]
    Y          = np.loadtxt('~/Workspace/datasets/KernelMethods/Assignment6/data/y.txt')
    test_idxs  = np.loadtxt('~/Workspace/datasets/KernelMethods/Assignment6/data/test_set.txt')
    train_idxs = np.loadtxt('~/Workspace/datasets/KernelMethods/Assignment6/data/train_set.txt')

    # get the linear combination weights
    d = get_alignment(Kernels, Y) 

    # calculate the ALIGNF kernel
    alignf = np.zeros_like(Kernels[0])
    for idx,kernel in enumerate(Kernels)
        np.sum(alignf, kernel * d[idx], out=alignf)

    kernel_dict = {'ALIGNF':alignf}

    for kernel_filename, kernel in zip(kernel_files, Kernels):
        kernel_name = kernel_filename.split(".")[0].split("/")[2]
        kernel_dict[kernel_name] = kernel
    
    # For each of the 7 Kernels
        # train 13 one-vs-all svm classifiers
        # Calculate the classification accuracy 

        
    
