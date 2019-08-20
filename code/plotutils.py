import warnings
warnings.filterwarnings('ignore')

from utils import *
from tapdynamics import *
from particlefilter import *

def generatedata(brain, theta, params, use_cuda, B, T, T_clip):

    """
    Generate input data
    """
    x, y = generate_trainingdata(theta, params, B, T, T_clip)[0:2]

    """
    Run brain on inputs
    """
    if use_cuda and torch.cuda.is_available():
        y = y.cuda()

    r_brain = brain(y)[0]

    y       = y[:,T_clip:].cpu().data.numpy()
    r_brain = r_brain[:,T_clip:].cpu().data.numpy()
    x       = x[:,T_clip:]

    """
    Reshape the data variables
    """
    y       = y.transpose(2,1,0)
    r_brain = r_brain.transpose(2,1,0)

    return y, x, r_brain


def loadbrain(fname, use_cuda):

    """
    Load the tap brain model
    """
    brain = torch.load(fname + '.pt')

    if use_cuda and torch.cuda.is_available():
        brain.cuda()
    else:
        brain.use_cuda = False

    """
    Load required data
    """
    with open(fname + '_params.pkl','rb') as f:
        theta, params = pickle.load(f)
    f.close()

    return brain, theta, params

def runparticlefilter(r_brain, y, theta, params):

    Ns, Ny      = params['Ns'], params['Ny']
    Np          = params['Np'] # No. of particles
    Nr, T, B    = r_brain.shape
    
    U = extractParams(theta, 18, Ns, Ny, Nr)[3]

    x_dec = np.zeros([Ns,T+1,B])    # decoded latent dynamics using ground truth parameters (tp)
    P_dec = np.zeros([Ns,Np,T+1,B]) # dynamics of individual particles
    r_dec = np.zeros([Nr,T,B])      # fit to measurements using ground truth parameters
    W_dec = np.zeros([Np,B])        # weights of each particles

    LL_dec = np.zeros([B])          # data log-likelihood

    for bi in range(B):
        LL_dec[bi], x_dec[:,:,bi], P_dec[:,:,:,bi], W_dec[:,bi] = particlefilter(r_brain[:,:,bi], y[:,:,bi], Np, params['Q_process'], params['Q_obs'], theta, params['nltype'])
        r_dec[:,:,bi] = np.dot(U, x_dec[:,1:,bi])

    return LL_dec, x_dec, r_dec, P_dec, W_dec



def runNegLL(theta, r, y, particles, weights, params, computegrad, alpha_J, alpha_G):
    C = 0
    B = r.shape[2]
    for b in range(B):
        C += NegLL(theta, r[...,b], y[...,b], particles[...,b], weights[...,b], params['Q_process'], params['Q_obs'], params['nltype'], computegrad, alpha_J, alpha_G)  
    return C
    