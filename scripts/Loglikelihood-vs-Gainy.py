import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../code')
from plotutils import *

trial       = int(sys.argv[1])      # trial number
EMIters     = int(sys.argv[2])      # No. of EM iters
Ns_hat      = int(sys.argv[3])      # assumed value of Ns
q_obs       = float(sys.argv[4])    # assumed observation noise variance
Gainy       = int(sys.argv[5])

q_process   = 1e-6 


"""
1. Load the data for ICA
"""
r_brain = np.load('../results/Experiment5/recordings/r_brain_ICA_' + str(Gainy) + '.npy')
y       = np.load('../results/Experiment5/recordings/y_ICA_' + str(Gainy) + '.npy')

Nr, T, B    = r_brain.shape
Ny          = y.shape[0]

"""
3. Use ICA to get initial estimate of embedding
"""
U_hat       = UhatICA(np.reshape(r_brain,[Nr,T*B],order='F').T, Ns_hat)[0]


"""
4. Initialize the rest of the parameters
"""
sparsity_J  = 0
Jtype       = 'nonferr'
self_coupling_on = 1

G_hat       = np.zeros([18])
J_hat       = Create_J(Ns_hat, sparsity_J, Jtype, self_coupling_on) 

if Ns_hat <= Ny:
    V_hat = np.linalg.svd(np.random.randn(Ns_hat,Ny), full_matrices=False)[2]
else:
    V_hat = np.linalg.svd(np.random.randn(Ns_hat,Ny), full_matrices=False)[0]

print('Ns_hat =', J_hat.shape[0])

"""
5. Load the data for EM
"""
r_brain = np.load('../results/Experiment5/recordings/r_brain_EM.npy')
y       = np.load('../results/Experiment5/recordings/y_EM.npy')

Nr, T, B    = r_brain.shape
Ny          = y.shape[0]

lam = np.array([0.25])

# Reshape the neural activity and measurements into batches x no. of vars x time
r_brain = r_brain.transpose(2,0,1)
y       = y.transpose(2,0,1)


"""
6. Convert data to torch tensors
"""
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype       = torch.float64

Q_process   = torch.tensor(q_process*np.eye(Ns_hat),device=device,dtype=dtype)
Q_obs       = torch.tensor(q_obs*np.eye(Nr),device=device,dtype=dtype)

P_process   = Q_process.inverse()
P_obs       = Q_obs.inverse()

r_brain     = torch.tensor(r_brain,device=device,dtype=dtype)
y           = torch.tensor(y,device=device,dtype=dtype)

lam         = torch.tensor(lam,device=device,dtype=dtype,requires_grad=False)
G_hat       = torch.tensor(G_hat,device=device,dtype=dtype,requires_grad=True)
J_hat       = torch.tensor(J_hat,device=device,dtype=dtype,requires_grad=True)
U_hat       = torch.tensor(U_hat,device=device,dtype=dtype,requires_grad=True)
V_hat       = torch.tensor(V_hat,device=device,dtype=dtype,requires_grad=True)


"""
7. Run PF-EM
"""

# Run the PF with initial estimates of the parameters
Np = 100 # No. of particles to use
with torch.no_grad():
    LL_hat, x_hat, P_hat, W_hat = particlefilter_torch(G_hat, J_hat, U_hat, V_hat, lam, r_brain, y, P_process, P_obs, Np)
    
print('Log Likelihood with initial parameters = %.3f' %(LL_hat.mean()))

lrate     = 2e-2
opt_params= [G_hat,J_hat,U_hat,V_hat]
T_st      = 10 # burn in time of the particle filter

LLVec     = []

optimizer = torch.optim.Adam(opt_params,lr=lrate, betas=(0.9, 0.999))


t_st = time.time() 

for epoch in range(EMIters):
    
    # zero-gradients at the start of each epoch
    optimizer.zero_grad() 
    
    # E-step
    C = Qfunction_torch(G_hat, J_hat, U_hat, V_hat, lam, r_brain[...,T_st:], y[...,T_st:], P_hat[...,T_st:], W_hat, P_process, P_obs)
    
    # M-step
    C.backward() 
    G_hat.grad[0], G_hat.grad[9] = 0, 0 # set gradient of G0 and G9 to zero
    optimizer.step()
    
    # Run PF again to get posterior for E-step
    with torch.no_grad():
        LL_hat, x_hat, P_hat, W_hat = particlefilter_torch(G_hat, J_hat, U_hat, V_hat, lam, r_brain, y, P_process, P_obs, Np)
        
    LLVec.append(LL_hat.mean())
    
    if (epoch+1) % 100 == 0:
        print('[%d] training loss: %.5f' %(epoch + 1, LL_hat.mean()))

t_en = time.time()

np.save('./Loglikelihood-vs-Gainy/LL_Gainy'+str(Gainy)+'_trial'+str(trial), np.array(LLVec))

print('Finished training')
print('time elapsed = %.2f s' %(t_en - t_st))

print('Likelihood post EM = %.3f' %(LL_hat.mean()))