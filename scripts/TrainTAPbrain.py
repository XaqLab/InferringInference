import sys
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../code')

from utils import *
from rnnmodel import *
from tapdynamics import *


def generateModelParameters(Ns, noise_seed):
    """
    Parameters and settings used to generate TAP dynamics
    """
    
    Nr = 2*Ns  # No. of neurons
    Ny = Ns+1  # No. of input variables

    # Noise covariances 
    Q_process = 0*np.eye(Ns) # process noise
    Q_obs = 0*np.eye(Nr)        # use zero noise in train inputs to RNN

    # Input setting
    gain_y = 25/np.sqrt(Ns) # gain for inputs y

    # Filter used for smoothing the input signals
    smoothing_filter = signal.hamming(5,sym=True) 
    smoothing_filter = smoothing_filter/sum(smoothing_filter)

    # TAP model parameters

    lam = np.array([0.25])  # low pass filtering constant for the TAP dynamics

    nltype = 'sigmoid' # external nonlinearity in TAP dynamics

    G = np.array([0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]) # message passing parameters of the TAP equation

    self_coupling_on = 1 # self coupling in J ON
    sparsity_J = 0.3     # sparsity in J 
    Jtype = 'nonferr'
    J = 3*Create_J(Ns, sparsity_J, Jtype, self_coupling_on) # Coupling matrix  

    U = 2*np.random.rand(Nr,Ns) # embedding matrix
    # U = np.eye(Nr)

    V = np.linalg.svd(np.random.randn(Ns,Ny), full_matrices=False)[2] # input embedding matrix

    # concatenate the parameters
    theta = np.concatenate([lam, G, JMatToVec(J), U.flatten('F'), V.flatten('F') ])

    params = {'Ny':Ny, 'Nr': Nr, 'Ns': Ns, 'Q_process': Q_process, 'Q_obs': Q_obs, 'nltype': nltype, 'Jtype':Jtype, 'sparsity_J': sparsity_J, 'self_coupling_on': self_coupling_on, 'gain_y':gain_y, 'smoothing_filter': smoothing_filter}

    return theta, params





def createbrain(N_input, N_hidden, N_output, use_cuda):
    """
    Create RNN module
    """
    brain = RNN(N_input, N_hidden, N_output, use_cuda)

    if use_cuda and torch.cuda.is_available():
        brain.cuda()

    return brain


def trainbrain(brain, y_train, y_val, r_train, r_val, NEpochs, T_clip, use_cuda, learningrate):

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(brain.parameters(),lr=learningrate, betas=(0.9, 0.999))


    if use_cuda and torch.cuda.is_available():
        y_train = y_train.cuda()
        r_train = r_train.cuda()
        y_val = y_val.cuda()
        r_val = r_val.cuda()

    t_st = time.time()

    train_loss, val_loss = np.zeros([NEpochs]), np.zeros([NEpochs])

      
    for epoch in range(NEpochs):
        
        optimizer.zero_grad() # zero-gradients at the start of each epoch
        
        # training data
        rhat_train = brain(y_train)[0]
        
        mse_train = loss_fn(r_train[:,T_clip:],rhat_train[:,T_clip:])
        
        mse_train.backward()  # do backprop to compute gradients 
        
        optimizer.step() # optimization step
        
        # validation data
        rhat_val = brain(y_val)[0]
        
        mse_val = loss_fn(r_val[:,T_clip:],rhat_val[:,T_clip:])
        
        # record loss
        train_loss[epoch] = mse_train.item()
        val_loss[epoch]   = mse_val.item()
        
        if epoch % 1000 == 999:
            print('[%d] training loss: %.5f' %(epoch + 1, mse_train.item()))

    print('Finished training')
    t_en = time.time()
    print('Time elapsed =', np.round(1000*(t_en - t_st))/1000, 's')

    return brain, train_loss, val_loss



def main():

    use_cuda = True

    # Input parameters
    Ns              = int(sys.argv[1]) # No. of variables
    noise_seed      = int(sys.argv[2])
    N_hidden        = int(sys.argv[3]) 
    NEpochs         = int(sys.argv[4])
    learningrate    = int(sys.argv[5])*1e-5

    print('noise_seed =',noise_seed)
    np.random.seed(seed=noise_seed)


    theta, params   = generateModelParameters(Ns, noise_seed)

    B_train, B_val  = 1000, 100 # No. of batches
    T_train, T_val  = 80, 80 # No. of time steps 
    T_clip          = 20 # No. of time steps that will be clipped; corresponds to rnn burn in time

    y_train, r_train    = generate_trainingdata(theta, params, B_train, T_train, T_clip)[1:]
    y_val, r_val        = generate_trainingdata(theta, params, B_train, T_train, T_clip)[1:]
 
    tapbrain = createbrain(params['Ny'], N_hidden, params['Nr'], use_cuda)

    print('learningrate =',learningrate)
    print('Training brain ... ')
    tapbrain, train_loss, val_loss = trainbrain(tapbrain, y_train, y_val, r_train, r_val, NEpochs, T_clip, use_cuda, learningrate)
    print('Training complete. Saving brain at')

    tapbrain.cpu()

    brain_name = '../data/brains/' + 'Ns_'+ str(Ns) + '_noiseseed_' + str(noise_seed)
    
    torch.save(tapbrain, brain_name + '.pt')

    params['train_loss'] = train_loss
    params['val_loss'] = val_loss

    print(brain_name + '.pt')

    with open(brain_name + '_params.pkl', 'wb') as f:  
        pickle.dump([theta, params], f)
    f.close()



if __name__ == "__main__":
    main()
