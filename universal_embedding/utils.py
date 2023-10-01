import json
import numpy as np




def normalize(a,axis=-1,order=2):
    '''Normalize descriptors (l2 normalization by default)
    '''
    l2 = np.linalg.norm(a, order, axis)
    l2[l2==0] = 1

    return a / np.expand_dims(l2, axis)



def apply_pca_whiten_and_normalize(X, m, P):
    '''Apply given learned pca whitening matrix to given descriptors after subtracting the learned mean.
    
    Normalizes (l2) as well

    '''
    X = np.dot(X-m, P)
    return normalize(X,axis = 1)



def estimate_pca_whiten_with_shrinkage(X, shrinkage=1.0, dimensions=None):
    '''
    Learn pca whitening with given shrinkage
    "dimensions" argument is the dimensions that we keep after the pca-whitening procedure
    shrinkage = 1 corresponds to pca whitening
    shrinkage = 0 corresponds to pca
    '''
    n,d = X.shape[0],X.shape[1]

    m = X.mean(axis=0, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc.T, Xc)
    Xcov = (Xcov + Xcov.T) / (2*n)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]   
    eigval = eigval[:dimensions]
    eigvec = eigvec[:,:dimensions]
    P = np.dot(np.linalg.inv(np.diag(np.power(eigval,0.5*shrinkage))), eigvec.T)

    return m,P.T



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)