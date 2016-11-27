import pdb
import pprint
from collections import OrderedDict
import numpy as np
import scipy as sp
from scipy import sparse
from sklearn import svm

def bag_of_words_frm_str(string,delim=" "):
    bag_of_words_str = {}
    for word in string.strip().split(delim):
        try:
            bag_of_words_str[word] += 1
        except KeyError:
            bag_of_words_str[word] = 1

    return bag_of_words_str

def get_bag_of_words(filename):
    
    bag_of_words = []
    words = {}

    with open(filename,"r") as f:
        data = f.readlines()
        for line in data:
            bow_line = bag_of_words_frm_str(line)
            bag_of_words.append(bow_line)
            words.update(bow_line)
            
    return bag_of_words, set(words) # creates a set of keys implicitly

def populate_dt_matrix(doc_term_mat, documents, common_word_set):
    common_word_set_indexed = dict(zip(common_word_set, range(len(common_word_set))))
    for idx, doc in enumerate(documents):
        for term,count in doc.items():
            try:
                doc_term_mat[idx,common_word_set_indexed[term]] = count
            except KeyError:
                print("Key {} in Doc but not in common_term_set !!".format(term))

if __name__ == "__main__":
    # get bag of words from train and test files
    Xtest_bow, Xtest_words = get_bag_of_words("../data/text_test.txt")
    Xtrain_bow, Xtrain_words = get_bag_of_words("../data/text_train.txt")

    y_test = np.loadtxt("../data/y_test.txt")
    y_train = np.loadtxt("../data/y_train.txt")

    # get intersection of set of words from test and train docs
    common_word_set = Xtrain_words.union(Xtest_words)
    num_terms = len(common_word_set)
    
    # create the document-term matrix
    Xtest_dt  = np.zeros((len(Xtest_bow ), num_terms))
    Xtrain_dt = np.zeros((len(Xtrain_bow), num_terms))

    populate_dt_matrix(Xtest_dt, Xtest_bow, common_word_set)
    populate_dt_matrix(Xtrain_dt, Xtrain_bow, common_word_set)

    _term_1_doc = np.vstack((Xtest_dt, Xtrain_dt))

    # Identify terms which occur more than once
    occur_gt1_term = _term_1_doc.astype(np.bool).sum(axis=0) > 1 

    # keep terms which occur more than once
    Xtest_dt  = Xtest_dt[:, occur_gt1_term]
    Xtrain_dt = Xtrain_dt[:, occur_gt1_term]

    # Normalize the train and test matrix
    # mean = np.atleast_2d(np.mean(Xtrain_dt, axis=1)).T
    # std = np.atleast_2d(np.std(Xtrain_dt, axis=1)).T
    # Xtrain_dt -= mean
    # Xtest_dt  -= mean
    # Xtrain_dt /= std
    # Xtest_dt  /= std
    
    kernels = {}
    kernels['Bag of Words'] = {'train': np.dot(Xtrain_dt, Xtrain_dt.T),
                     'train_test': np.dot(Xtrain_dt, Xtest_dt.T)
                    }
    print('BOW Kernel Done..')
    # Document Frequency of terms
    nt = Xtrain_dt.astype(np.bool).sum(axis=0)
    # Number of documents
    idf = np.log(1.0 + Xtrain_dt.shape[0]/(nt+10**-5))

    Xtest_tfidf = Xtest_dt * idf
    Xtrain_tfidf = Xtrain_dt * idf

    kernels['TF-IDF'] = {'train': np.dot(Xtrain_tfidf, Xtrain_tfidf.T),
                       'train_test': np.dot(Xtrain_tfidf, Xtest_tfidf.T)}
    print('TFIDF Kernel Done..')

    Cov_train = (1.0/Xtrain_dt.shape[0]) * np.dot(Xtrain_dt.T, Xtrain_dt)
    Cov_test  = (1.0/Xtest_dt.shape[0]) * np.dot(Xtest_dt.T, Xtest_dt)

    kernels['Generalized Vector Space'] = {'train': np.dot(np.dot(Xtrain_dt,Cov_train), Xtrain_dt.T),
                    'train_test': np.dot(np.dot(Xtrain_dt, Cov_test), Xtest_dt.T)}
    print('Vector space kernel Done..')

    U,s,V = np.linalg.svd(Xtrain_dt.T) # V is the Eigen Value of covariance matrix  
    # for k in [1, 2, 5, 10]:
    for k in [1, 50, 500, 1000]:
        uNew = U[:,:k]
        uuT = np.dot(uNew, uNew.T)
        kernels['Latent Semantic Kernel with K-{}'.format(k)] = {'train': np.dot(np.dot(Xtrain_dt,uuT), Xtrain_dt.T),
                        'train_test': np.dot(np.dot(Xtrain_dt, uuT), Xtest_dt.T)}
        print('Latent Semantic kernel for k = {} Done..'.format(k))

    # U,s,V = np.linalg.svd(Xtrain_dt) # V is the Eigen Value of covariance matrix  
    # for k in [1, 2, 5, 10]:
    #     vNew = V[:,:k]
    #     vvT = np.dot(vNew, vNew.T)
    #     kernels['Latent Semantic Kernel with K-{}'.format(k)] = {'train': np.dot(np.dot(Xtrain_dt,vvT), Xtrain_dt.T),
    #                     'train_test': np.dot(np.dot(Xtrain_dt, vvT), Xtest_dt.T)}
    #     print('Latent Semantic kernel for k = {} Done..'.format(k))
    
    # U,s,V = sparse.linalg.svds(sparse.csr_matrix(Xtrain_dt),k=50) # V is the Eigen Value of covariance matrix  
    # for k in [1, 2, 5, 10, 50]:
    #     pdb.set_trace()
    #     uNew = U[:,:k]
    #     uuT = np.dot(uNew, uNew.T)
    #     kernels['ls-{}'.format(k)] = {'train': np.dot(np.dot(Xtrain_dt,uuT), Xtrain_dt.T),
    #                     'train_test': np.dot(np.dot(Xtrain_dt, uuT), Xtest_dt.T)}
    #     print('Latent Semantic kernel for k = {} Done..'.format(k))



    # C = 10.0**np.arange(-1,3)
    C = 10.0**np.arange(-2,3)
    results = {}
    for kernel,val in kernels.items():
        Ktrain, Ktrain_test = val['train'], val['train_test']
        print("=== {} ===".format(kernel))
        for c in C:
            model = svm.SVC(C=c, kernel='precomputed')
            model.fit(Ktrain, y_train)
            y_predict = model.predict(Ktrain_test.T)
            results[(kernel, c)] = np.mean(y_predict==y_test)
            print("C = {} Accuracy = {}".format(c, results[(kernel, c)]))
    # pprint.pprint(results)
    pdb.set_trace()
