import sys
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../code')

from utils import *
from rnnmodel import *
from tapdynamics import *
from particlefilter import *


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


def initparameters(r_brain, theta, params):
    
    Ns, Ny      = params['Ns'], params['Ny']
    Nr, T, B    = r_brain.shape

    # extract true parameters
    lam, G, J, U, V = extractParams(theta, 18, Ns, Ny, Nr)
    
    # Use ICA to get initial estimate of the embedding
    U_hat   = UhatICA(np.reshape(r_brain,[Nr,T*B], order='F').T, Ns)[0]

    # Estimate the permutation matrix P (need this to resolve permutation ambiguity of ICA)
    PermMat = EstimatePermutation_ICA(U,U_hat)

    lam_hat = np.array([lam])                       
    #G_hat   = 0.1*np.random.randn(18)
    G_hat   = np.random.randn(18)
    G_hat[0]= 0 #force this condition
    G_hat[9]= 0 #force this condition          
    J_hat   = Create_J(Ns, params['sparsity_J'], 'nonferr', params['self_coupling_on']) 
    V_hat   = np.linalg.svd(np.random.randn(Ns,Ny), full_matrices=False)[2]

    theta_hat = np.concatenate([lam_hat, G_hat, JMatToVec(J_hat), U_hat.flatten('F'), V_hat.flatten('F') ])
    return theta_hat


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


def runPFEM(r_brain, y, theta, params,  EMIters, iter_updateU, alpha_J, alpha_G):
    """
    Run the particle EM algorithm
    """
    Ns, Ny      = params['Ns'], params['Ny']
    Np          = params['Np'] # No. of particles
    Nr, T, B    = r_brain.shape

    # Run PF once with initial value of parameters
    P_hat, W_hat = runparticlefilter(r_brain, y, theta, params)[3:]

    # Run the PF-EM on mini-batches of the entire dataset. For now, each mini-batch comprises just one individual session.
    idx     = np.random.randint(B)
    r_b     = r_brain[:,:,idx]      # pick the observations for the mini batch
    y_b     = y[:,:,idx]            # pick the input signals for the mini batch 
    P_b     = P_hat[:,:,:,idx]
    W_b     = W_hat[:,idx]

    LL_Vec          = np.zeros([EMIters])  # record how the log-likelihood changes with iterations
    MStepMaxIter    = 10               # Maximum no. of iterations used by the optimizer in the M step
    computegrad     = np.array([1,1,0,1,0], dtype=int) # Flags which indicate which variables are updated in the order: G, J, U, V, lam

    iter_print      = 20

    # We keep Uhat fixed for the first iter_updateU iterations and update the rest. After that, we update all the parameters together.

    for iterem in range(EMIters):
        
        if iterem == iter_updateU:
            computegrad = np.array([1,1,1,1,0], dtype=int)

        if (iterem+1) % iter_print == 0:
            print('iterem =', iterem + 1)  
        
        MStep = optimize.minimize(NegLL, theta, args = (r_b, y_b, P_b, W_b, params['Q_process'], params['Q_obs'], params['nltype'], computegrad, alpha_J, alpha_G), method='BFGS', jac = NegLL_D, options={'disp': False,'maxiter':MStepMaxIter})
        theta = MStep.x
            
        # E step: Pick a new batch and run the particle filter with the updated parameters    
        idx     = np.random.randint(B)
        r_b     = r_brain[:,:,idx] 
        y_b     = y[:,:,idx]
        
        LL_Vec[iterem], x_b, P_b, W_b = particlefilter(r_b, y_b, params['Np'], params['Q_process'], params['Q_obs'], theta, params['nltype'])

    return theta, LL_Vec


def main():

    use_cuda = True    
    
    # Input parameters
    fname           = sys.argv[1]       # No. of variables
    noise_seed      = int(sys.argv[2])
    EMIters         = int(sys.argv[3])  # No. of EM iters
    iter_updateU    = int(sys.argv[4])  # No. of iters after which to update U
    alpha_J         = int(sys.argv[5])  # L1 regularization coefficient for J
    alpha_G         = int(sys.argv[6])  # L1 regularization coefficient for G

    np.random.seed(seed=noise_seed)
    torch.manual_seed(noise_seed)

    print('loading brain')
    tapbrain, theta, params = loadbrain('../data/brains/'+ fname, use_cuda)

    print('record brain activity')
    B, T, T_clip    = 1, 2500, 20 #No. of batches, No. of time steps, No. of time steps to clip (RNN burn in time)
    y, x, r_brain   = generatedata(tapbrain, theta, params, use_cuda, B, T, T_clip)

    print('Initialize PF_EM')
    theta_init  = initparameters(r_brain, theta, params)
    theta_hat   = theta_init*1.0

    # Retain only the required subset of data for PF-EM
    TTotal  = 500 # Total no. of time steps to use for analysis
    T       = TTotal//B
    y, x, r_brain = y[:,0:T], x[:,0:T+1], r_brain[:,0:T]

    # update parameters for the particle filter
    params['Q_process'] = 1e-5*np.eye(params['Ns'])     # process noise
    params['Q_obs']     = 5e-4*np.eye(params['Nr'])     # measurement noise
    params['Np']        = 100                           # No. of particles
    
    # run PF with ground truth parameters
    LL_tp, x_tp, r_tp = runparticlefilter(r_brain, y, theta, params)[0:3]
    print('Log likelihood with true params = ', LL_tp.mean())

    # run PF with initial values of parameters
    LL_init, x_init, r_init = runparticlefilter(r_brain, y, theta_init, params)[0:3]
    print('Log likelihood pre PF-EM = ', LL_init.mean())

    # run PF-EM
    theta_hat, LL_Vec = runPFEM(r_brain, y, theta_hat, params, EMIters, iter_updateU, alpha_J, alpha_G)

    # run PF with estimated values of parameters
    LL_hat, x_hat, r_hat = runparticlefilter(r_brain, y, theta_hat, params)[0:3]
    print('Log likelihood post PF-EM = ', LL_hat.mean())

    # save variables
    true_estimates  = {'LL_tp':LL_tp, 'x_tp':x_tp, 'r_tp':r_tp}
    init_estimates  = {'theta_init':theta_init, 'LL_init':LL_init, 'x_init':x_init, 'r_init':r_init}
    final_estimates = {'theta_hat':theta_hat, 'LL_hat':LL_hat, 'x_hat':x_hat, 'r_hat':r_hat, 'LL_Vec':LL_Vec}
    with open('../data/estimates/' + fname + '_' + str(alpha_J) + '_' + str(alpha_G) + '.pkl', 'wb') as f:  
        pickle.dump([r_brain, x, y, theta, params, true_estimates, init_estimates, final_estimates], f)
    f.close()




if __name__ == "__main__":
    main()
