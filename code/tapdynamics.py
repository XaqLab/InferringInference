import numpy as np
import torch
from scipy import signal
from utils import extractParams, nonlinearity


def Create_J(Nx, sp, Jtype, SelfCoupling):
    
    """
    Generate a sparse, symmetric coupling matrix with desired kind of interactions

    Inputs: 
    Nx    : No. of x's
    sp    : degree of sparsity of J
    Jtype : coupling type - ferromagnetic (all positive), antiferr (all negative), nonferr (mixed)
    SelfCoupling: determines if J matrix has self coupling or not

    Output
    J     : coupling matrix
    """

    # Create the mask for zeros
    H = np.random.rand(Nx,Nx)
    H = np.tril(H,k=-1)
    H[H < sp] = 0
    H[H >= sp] = 1
    
    if (SelfCoupling == 1):
        H = H + H.T + np.eye(Nx)
    else:
        H = H + H.T
        
    # Create full coupling matrix with required kind of interaction
    
    if Jtype == 'ferr':
        J = np.tril(np.random.rand(Nx,Nx),-1)
        J = J + J.T + np.diag(np.random.rand(Nx))
        J = J/np.sqrt(Nx)
    elif Jtype == 'antiferr':
        J = -np.tril(np.random.rand(Nx,Nx),-1)
        J = J + J.T + np.diag(np.random.rand(Nx))
        J = J/np.sqrt(Nx)
    else:
        J = np.tril(0.5*np.random.randn(Nx,Nx),-1)
        J = J + J.T + np.diag(0.5*np.random.randn(Nx))
        J = J/np.sqrt(Nx)
        
    # Apply mask
    if sp != 0:
        J = J*H
        
    return J


def generateBroadH(Nx,T,Th,scaling):
    """
    Function to generate h(t), the input to the TAP dynamics
    Modeling h(t) such that it stays constant for every Nh time steps.
    """    

    # First generate only T/Nh independent values of h
    shape = 1 # gamma shape parameter
    Lh = T//Th + 1*(T%Th != 0)
    gsmScale = np.random.gamma(shape,scaling,(Nx,Lh))
    hInd = gsmScale*np.random.randn(Nx,Lh)
    hMat = np.zeros([Nx,T])

    # Then repeat each independent h for Nh time steps
    for t in range(T):
        hMat[:,t] = hInd[:,t//Th]
        
    return hMat


     
    
def runTAP(x0, hMat, Qpr, Qobs, theta, nltype):

    """
    % Function that generates the TAP dynamics

    % Inputs: 
    % x0    : latent variables at time t = 0
    % hMat  : of size Nx x T, specifies inputs h(t) for t = 1,..,T
    % lam   : low pass fitlering constant for TAP dynamics
    % Qpr   : covariance of process noise
    % Qobs  : covariance of measurement noise
    % U     : embedding matrix from latent space to neural activity
    % V     : emedding matrix from input space to latent variable space
    % J     : coupling matrix of the underlying distribution
    % G     : global hyperparameters

    % Outputs: 
    % xMat  : latent variables 
    % rMat  : neural activity. r = Ux + noise
    """

    Nh, T = hMat.shape # input dimensions, no. of time steps
    Nx = Qpr.shape[0]  # latent dimensions
    Nr = Qobs.shape[0] # output dimensions

    lG = 18 # hard coded for now
    lam, G, J, U, V = extractParams(theta, lG, Nx, Nh, Nr)

    x = x0 # initial value of x

    xMat = np.zeros([Nx,T+1])
    xMat[:,0] = x0

    J2 = J**2

    for tt in range(T):  
        
        ht = hMat[:,tt]

        x2      = x**2
        J1      = np.dot(J,np.ones([Nx]))
        Jx      = np.dot(J,x)
        Jx2     = np.dot(J,x2)
        J21     = np.dot(J2,np.ones([Nx]))
        J2x     = np.dot(J2,x)
        J2x2    = np.dot(J2,x2)

        argf = np.dot(V,ht) + G[0]*J1 + G[1]*Jx + G[2]*Jx2 + G[9]*J21 + G[10]*J2x + G[11]*J2x2 + x*( G[3]*J1 + G[4]*Jx + G[5]*Jx2 + G[12]*J21 + G[13]*J2x + G[14]*J2x2 ) + x2*(G[6]*J1 + G[7]*Jx + G[8]*Jx2 + G[15]*J21 + G[16]*J2x + G[17]*J2x2)
        
        TAPFn = nonlinearity(argf, nltype)[0]
        xnew = (1-lam)*x + lam*TAPFn + np.random.multivariate_normal(np.zeros(Nx),Qpr)
        xMat[:,tt+1] = xnew
        x = xnew

    rMat = np.dot(U,xMat[:,1:]) + np.random.multivariate_normal(np.zeros(Nr),Qobs,T).T  #Adding independent observation noise to each time step

    return xMat, rMat


def generate_trainingdata(theta, params, B, T, T_clip):
    """
    Generate training and validation data
    """
    Ns = params['Ns']
    Ny = params['Ny']
    Nr = params['Nr']
    Q_process = params['Q_process']
    Q_obs = params['Q_obs']
    nltype = params['nltype']
    gain_y = params['gain_y']
    smoothing_filter = params['smoothing_filter']

    T_const = np.random.randint(low=2,high=20,size=(B))

    y = np.zeros([Ny, T + T_clip, B])

    # Initial values of latent dynamics
    x0 = np.random.rand(Ns,B)

    # Initialize arrays to save dynamics
    x = np.zeros([Ns, T + T_clip + 1, B])
    r = np.zeros([Nr, T + T_clip , B])

    for bi in range(B):
        y[:,:,bi] = signal.filtfilt( smoothing_filter, 1, generateBroadH(Ny, T + T_clip, T_const[bi], gain_y) )
        x[:,:,bi], r[:,:,bi] = runTAP(x0[:,bi], y[:,:,bi], Q_process, Q_obs, theta, nltype)


    """
    Convert ground truth dynamics data to torch tensors
    """    
    y = torch.tensor(y.transpose(2,1,0), dtype=torch.float32) # input signal
    r = torch.tensor(r.transpose(2,1,0), dtype=torch.float32) # target neural activity

    return x, y, r