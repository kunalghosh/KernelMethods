import pdb
import numpy as np
from sklearn import svm
from get_alignment import get_alignment

if __name__ == '__main__':

    # Load data
    path_prefix ='/Users/kunal/Workspace/datasets/KernelMethods/Assignment6/data' 

    kernel_files = [path_prefix + '/K_exp.txt',
                    path_prefix + '/K_mgi.txt',
                    path_prefix + '/K_mpi.txt',
                    path_prefix + '/K_pfamdom.txt',
                    path_prefix + '/K_sw.txt',
                    path_prefix + '/K_tap.txt']

    print("Loading data...")
    Kernels    = [np.loadtxt(file) for file in kernel_files]
    Y          = np.loadtxt(path_prefix + '/y.txt').astype(np.int32)
    test_idxs  = np.loadtxt(path_prefix + '/test_set.txt').astype(np.int32)-1
    train_idxs = np.loadtxt(path_prefix + '/train_set.txt').astype(np.int32)-1
    print("Data Loaded...")

    # get the linear combination weights
    d = get_alignment(Kernels, Y) 
    print("Got the linear combination weights...")

    # calculate the ALIGNF kernel
    alignf = np.zeros_like(Kernels[0])
    for idx,kernel in enumerate(Kernels):
        # np.sum(alignf, kernel * d[idx], out=alignf)
        alignf += kernel * d[idx]
    print("Calculated the ALIGNF kernel...")

    kernel_dict = {'ALIGNF':alignf}

    for kernel_filename, kernel in zip(kernel_files, Kernels):
        kernel_name = kernel_filename.split(".")[0].split("/")[-1]
        kernel_dict[kernel_name] = kernel

    Accuracies = {}
    for name, kernel in kernel_dict.items():
        Accuracies[name] = []
        # For each of the 7 Kernels
        ktrain = kernel[train_idxs,:][:,train_idxs]
        ktest_train = kernel[test_idxs,:][:,train_idxs] 
        for col_idx in range(Y.shape[1]):
           model = svm.SVC(C=1, kernel='precomputed')
           model.fit(ktrain, Y[train_idxs,col_idx])
           y_predict = model.predict(ktest_train)
           Accuracies[name].append(np.mean(y_predict == Y[test_idxs,col_idx]))
           # train 13 one-vs-all svm classifiers
           # Calculate the classification accuracy 
        print("{} : {} ".format(name, Accuracies[name]))
