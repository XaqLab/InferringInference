import time
import numpy as np
from scipy import signal
from scipy import optimize
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import pickle
import torch


def nonlinearity(x,nltype):

    if nltype == 'sigmoid':
        y = 1/(1 + np.exp(-x))
        dy = y*(1-y)
    elif nltype == 'expsqrt':
        y   = np.sqrt(np.log(1 + np.exp(x)))
        dy  = np.exp(x)/(y*(1+np.exp(x)))
    elif nltype == 'dgauss':
        y   = .5 + x*np.exp(-x**2)
        dy  = np.exp(-x**2)*(1-2*x**2)
    elif nltype == 'xcauchytanh':
        y   = .5 + x/(1+x**2) + .05*np.tanh(x)
        dy  = 1/(1 + x**2) + x/((1 + x**2)**2) + 0.05*1/(np.cosh(x)**2)
    else:
        print('Nonlinearity unknown')
        
    return y, dy


def JVecToMat(JVec,Nx):
    JMat = np.zeros([Nx,Nx])
    for kk in range(Nx):
        JMat[kk:,kk] = JVec[0:Nx-kk]
        JMat[kk,kk:] = JVec[0:Nx-kk]
        JVec = np.delete(JVec,np.arange(Nx-kk))
        
    return JMat

def JMatToVec(JMat):
    Nx = np.shape(JMat)[0]
    JVec = np.zeros([1])
    
    for kk in range(Nx):
        JVec = np.concatenate((JVec,JMat[kk:,kk].flatten()),axis=0)
        
    JVec = JVec[1:]
        
    return JVec


def extractParams(theta, lG, Nx, Nh, Nr):
    # extract the parameters
    NJ = np.int(Nx*(Nx+1)/2)
    
    lam = theta[0]
    G = theta[1:1+lG]
    JVec = theta[1+lG:1+lG+NJ]
    J = JVecToMat(JVec,Nx)
    U = np.reshape(theta[1+lG+NJ:1+lG+NJ+Nr*Nx],[Nr,Nx],'F')
    V = np.reshape(theta[1+lG+NJ+Nr*Nx:],[Nx,Nh],'F')
    
    return lam, G, J, U, V


def JVecToMat_torch(JVec,Nx,device,dtype):
    JMat = torch.zeros((Nx,Nx),device=device,dtype=dtype)
    for kk in range(Nx):
        JMat[kk:,kk] = JVec[0:Nx-kk]
        JMat[kk,kk:] = JVec[0:Nx-kk]
        JVec = JVec[Nx-kk:]
        
    return JMat

def JMatToVec_torch(JMat,device,dtype):
    Nx = JMat.shape[0]
    JVec = []
    for kk in range(Nx):
        JVec.append(JMat[kk:,kk].unsqueeze(1))
        
    return torch.cat(JVec).squeeze()


def extractParams_torch(theta, lG, Nx, Nh, Nr, device, dtype):
    # extract the parameters
    NJ = Nx*(Nx+1)//2
    
    lam = theta[0]
    G = theta[1:1+lG]
    JVec = theta[1+lG:1+lG+NJ]
    J = JVecToMat_torch(JVec,Nx,device,dtype)
    U = theta[1+lG+NJ:1+lG+NJ+Nr*Nx].view(Nr,Nx)
    V = theta[1+lG+NJ+Nr*Nx:].view(Nx,Nh)
    # U = np.reshape(theta[1+lG+NJ:1+lG+NJ+Nr*Nx],[Nr,Nx],'F')
    # V = np.reshape(theta[1+lG+NJ+Nr*Nx:],[Nx,Nh],'F')
    
    return lam, G, J, U, V
        


def UhatICA(R, Nx):
    """
    Function to recover initial estimate of the embedding matrix U
    from the latent dynamics using ICA
    """

    ica = FastICA(n_components=Nx, algorithm='deflation',fun='cube')
    # R = np.reshape(rMatFull,[Nr,T*Ns],order='F').T
    Xe = ica.fit_transform(R)  # Reconstruct signals
    Uhat = ica.mixing_  # Get estimated mixing matrix
    m = ica.mean_

    Xe = Xe + np.dot(np.linalg.pinv(Uhat),m)

    minx = np.min(Xe,axis=0)
    maxx = np.max(Xe,axis=0)
    DW = np.zeros([Nx])
    for ii in range(Nx):
        if abs(minx[ii]) > abs(maxx[ii]):
            DW[ii] = minx[ii]
        else:
            DW[ii] = maxx[ii]

    Uhat = Uhat*DW
    Xe = Xe/DW
    
    return Uhat, Xe.T


def EstimatePermutation_ICA(U,U_1):
    """
    Function that estimates the permutation matrix for ICA estimate of the embedding U
    Inputs:
        U       ground truth embedding
        U_1     ICA estimate of embedding
    Outputs:
        P       permutation matrix such that U = U_1 x P
    """
    Nx = U.shape[1]
    P = np.zeros([Nx,Nx])

    for i in range(Nx):
        err = np.sum((np.expand_dims(U_1[:,i],1) - U)**2, axis=0)
        idx = np.argsort(err)
          
        if i == 0:
            taken = np.array([idx[0]])
            P[i,idx[0]] = 1
        else:
            k = 0
            while np.intersect1d(idx[k], taken).shape[0] != 0:
                k += 1
                
            P[i,idx[k]] = 1
            taken = np.append(taken, idx[k])

    return P



    
def resampleSystematic(w, N):
    """
    % [ indx ] = resampleSystematic( w, N)
    % Systematic resampling method for particle filtering. 
    % Author: Tiancheng Li,Ref:
    % T. Li, M. Bolic, P. Djuric, Resampling methods for particle filtering, 
    % submit to IEEE Signal Processing Magazine, August 2013
    % Input:
    %       w    the input weight sequence 
    %       N    the desired length of the output sequence(i.e. the desired number of resampled particles)
    % Output:
    %       indx the resampled index according to the weight sequence
    """

    M = len(w)
    w = w/sum(w)
    Q = np.cumsum(w)
    indx = np.zeros([N],dtype=int)
    T = np.linspace(0,1-1/N,N) + np.random.rand(1)/N;

    i = 0
    j = 0
    
    while (i<N) and (j<M):
        while Q[j] < T[i]:
            j = j + 1

        indx[i] = j
        i = i + 1
        
    return indx


def resampleSystematic_torch(w, N, device, dtype):
    """
    torch version of resample systematic
    % [ indx ] = resampleSystematic( w, N)
    % Systematic resampling method for particle filtering. 
    % Author: Tiancheng Li,Ref:
    % T. Li, M. Bolic, P. Djuric, Resampling methods for particle filtering, 
    % submit to IEEE Signal Processing Magazine, August 2013
    % Input:
    %       w    the input weight sequence 
    %       N    the desired length of the output sequence(i.e. the desired number of resampled particles)
    % Output:
    %       indx the resampled index according to the weight sequence
    """

    M       = len(w)
    w       = w/torch.sum(w)
    Q       = torch.cumsum(w,dim=0)
    # indx    = torch.zeros([N],device=device,dtype=dtype)
    T       = torch.linspace(0,1-1/N,N,device=device,dtype=dtype) + torch.rand(1,device=device,dtype=dtype)/N
    indx    = []

    i = 0
    j = 0
    
    while (i<N) and (j<M):
        while Q[j] < T[i]:
            j = j + 1

        indx.append(j)
        i = i + 1
        
    return indx