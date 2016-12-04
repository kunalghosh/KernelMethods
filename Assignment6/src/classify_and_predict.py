import pdb
from collections import OrderedDict as dict
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

    d_all = dict()
    # get the linear combination weights
    for idx, y in enumerate(Y.T):
        # pdb.set_trace()
        d = get_alignment(Kernels, np.atleast_2d(y).T) 
        d_all[idx] = d
        print("Got weights for functional property {}".format(idx))
    print("Got the linear combination weights...")

    # calculate the ALIGNF kernel
    kern_shape = Kernels[0].shape
    alignf = np.zeros((Y.shape[1], kern_shape[0], kern_shape[1]))
    for yi in range(Y.shape[1]):
        for idx,kernel in enumerate(Kernels):
            # np.sum(alignf, kernel * d[idx], out=alignf)
            alignf[yi] += kernel * d_all[yi][idx]
        print("Calculated the ALIGNF kernel ... {}".format(yi))

    kernel_dict = dict({'ALIGNF':alignf})

    for kernel_filename, kernel in zip(kernel_files, Kernels):
        kernel_name = kernel_filename.split(".")[0].split("/")[-1]
        kernel_dict[kernel_name] = kernel

    print("\nKernels are in order:")
    print([key for key in kernel_dict])


    # Accuracies = {}
    # for name, kernel in kernel_dict.items():
    #     Accuracies[name] = []
    #     # For each of the 7 Kernels
    #     ktrain = kernel[train_idxs,:][:,train_idxs]
    #     ktest_train = kernel[test_idxs,:][:,train_idxs] 
    #     for col_idx in range(Y.shape[1]):
    #        model = svm.SVC(C=1, kernel='precomputed')
    #        model.fit(ktrain, Y[train_idxs,col_idx])
    #        y_predict = model.predict(ktest_train)
    #        Accuracies[name].append(np.mean(y_predict == Y[test_idxs,col_idx]))
    #        # train 13 one-vs-all svm classifiers
    #        # Calculate the classification accuracy 
    #     print("{} : {} ".format(name, Accuracies[name]))

    Accuracies = dict()
    for col_idx in range(Y.shape[1]):
        Accuracies[col_idx] = []
        # For each of the 7 Kernels
        for name, kernel in kernel_dict.items():
            if 'ALIGNF' == name:
                ktrain      = kernel[col_idx][train_idxs,:][:,train_idxs]
                ktest_train = kernel[col_idx][test_idxs,:][:,train_idxs] 
            else:
                ktrain      = kernel[train_idxs,:][:,train_idxs]
                ktest_train = kernel[test_idxs,:][:,train_idxs] 

            model = svm.SVC(C=1, kernel='precomputed')
            model.fit(ktrain, Y[train_idxs,col_idx])
            y_predict = model.predict(ktest_train)
            Accuracies[col_idx].append(np.mean(y_predict == Y[test_idxs,col_idx]))
            # train 13 one-vs-all svm classifiers
            # Calculate the classification accuracy 
        print("{} : {} ".format(col_idx, Accuracies[col_idx]))
