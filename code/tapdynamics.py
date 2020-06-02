import numpy as np
import torch
from scipy import signal
from utils import extractParams, nonlinearity


def Create_J(Ns, sparsity, Jtype, selfcoupling):
	
	"""
	Generate a sparse, symmetric coupling matrix with desired kind of interactions

	Inputs: 
	Ns          : No. of latent variables
	sparsity    : degree of sparsity of J
	Jtype       : interaction type: ferr (all positive), antiferr (all negative), nonferr (mixed)
	selfcoupling: determines if J matrix has self coupling or not

	Output
	J : interaction matrix
	"""

	# Create the sparse
	mask = np.tril((np.random.rand(Ns,Ns) >= sparsity)*1.0,k=-1)
	mask += mask.T + np.eye(Ns)*(selfcoupling == 1) 

	# Create full coupling matrix with required kind of interaction
	
	if Jtype == 'ferr':
		J = np.tril(np.random.rand(Ns,Ns),-1)
		J += J.T + np.diag(np.random.rand(Ns))
	   
	elif Jtype == 'antiferr':
		J = -np.tril(np.random.rand(Ns,Ns),-1)
		J += J.T - np.diag(np.random.rand(Ns))
		
	else:
		J = np.tril(0.5*np.random.randn(Ns,Ns),-1)
		J += J.T + np.diag(0.5*np.random.randn(Ns))
		
	# Apply mask, scale and return
	return J*mask/np.sqrt(Ns)


def generateBroadH(Ny,T,dT,scaling):

	"""
	Function to generate y(t), the input to the TAP dynamics
	Modeling y(t) such that it stays constant for every dT time steps.
	"""    

	# First generate only T/dT independent values of y
	shape   = 1 # gamma shape parameter
	L       = T//dT + 1*(T%dT != 0)
	gammma  = np.random.gamma(shape,scaling,(Ny,L))
	yInd    = gammma*np.random.randn(Ny,L)
	yMat = np.zeros([Ny,T])

	# Then repeat each independent h for Nh time steps
	for t in range(T):
		yMat[:,t] = yInd[:,t//dT]
		
	return yMat

	
def runTAP(x0, yMat, Qpr, Qobs, theta, nltype):

	"""
	Function that generates the TAP dynamics

	Inputs: 
	x0 	  : initial value of xt, of size Nx x B
	yMat  : of size B x Ny x T-1, specifies inputs y(t) for t = 0,..,T-2
	Qpr   : covariance of process noise
	Qobs  : covariance of observation noise
	theta : parameters of the TAP dynamics
	lam   : low pass fitlering constant for TAP dynamics
	U     : embedding matrix from latent space to neural activity
	V     : emedding matrix from input space to latent variable space
	J     : coupling matrix of the underlying distribution
	G     : global hyperparameters
	nltype: outer nonlinearity in TAP dynamics

	Outputs: 
	xMat  : latent variables of shape B x Ns x T
	"""
	B  = yMat.shape[0] 		# no. of batches 
	Ny = yMat.shape[1] 		# no. of input dimensions
	T  = yMat.shape[2] + 1 	# no. of time steps
	Ns = Qpr.shape[0]  		# no. of latent dimensions
	Nr = Qobs.shape[0] 		# no. of output dimensions

	lam, G, J, U, V = extractParams(theta, 18, Ns, Ny, Nr) # G has 18 parameters for now

	xMat 	= []
	#x 		= np.random.rand(Ns,B)
	x 		= x0
	xMat.append(x)			

	J2   = J**2

	for t in range(1,T):  
		
		y       = yMat[...,t-1].T # y should have shape Ny x B

		x2      = x**2
		J1      = np.expand_dims(np.dot(J,np.ones([Ns])),1)
		Jx      = np.dot(J,x)
		Jx2     = np.dot(J,x2)
		J21     = np.expand_dims(np.dot(J2,np.ones([Ns])),1)
		J2x     = np.dot(J2,x)
		J2x2    = np.dot(J2,x2)

		argf    = np.dot(V,y) + G[0]*J1 + G[1]*Jx + G[2]*Jx2 + G[9]*J21 + G[10]*J2x + G[11]*J2x2 + x*( G[3]*J1 + G[4]*Jx + G[5]*Jx2 + G[12]*J21 + G[13]*J2x + G[14]*J2x2 ) + x2*(G[6]*J1 + G[7]*Jx + G[8]*Jx2 + G[15]*J21 + G[16]*J2x + G[17]*J2x2)
		xnew    = (1-lam)*x + lam*nonlinearity(argf, nltype)[0] 
		xnew    += np.random.multivariate_normal(np.zeros(Ns),Qpr,B).T

		xMat.append(xnew)
		x 		= xnew

	return np.array(xMat).transpose(2,1,0)


def generate_Input(modelparameters, B, T, T_low, T_high, yG_low, yG_high):

	"""
	Function that generates the inputs to the TAP brain
	"""
	Ns 			= modelparameters['Ns']
	Ny 			= modelparameters['Ny']
	smoother    = modelparameters['smoothing_filter']

	y = []

	for b in range(B):
		T_const = np.random.randint(low=T_low,high=T_high)
		y_gain  = np.random.randint(low=yG_low,high=yG_high) if (yG_low < yG_high) else yG_low
		y_gain  = y_gain/np.sqrt(Ns)
		y_b     = signal.filtfilt(smoother, 1, generateBroadH(Ny, T-1, T_const, y_gain))
		y.append(y_b)		

	return np.array(y)


def generate_TAPdynamics(theta, modelparameters, B, T, T_low, T_high, yG_low, yG_high):
	
	"""
	Function that generates the TAP dynamics
	Inputs: 
	theta : TAP dynamics parameters
	modelparameters 
	B : no. of batches of data
	T : no. of time steps in each batch
	T_low to T_high: range of time period for which the input signal is held constant
	yG_low to y_high: range of input signal gain

	Outputs: 
	y  : input signal           - shape B x Ny x T
	x  : latent dynamics        - shape B x Ns x T
	r  : linear embedding of x  - shape B x Nr x T

	"""

	Ns 			= modelparameters['Ns']
	Ny 			= modelparameters['Ny']
	Nr 			= modelparameters['Nr']
	Q_process   = modelparameters['Q_process']
	Q_obs       = modelparameters['Q_obs']
	nltype      = modelparameters['nltype']

	U = extractParams(theta, 18, Ns, Ny, Nr)[3] # embedding matrix

	y = generate_Input(modelparameters, B, T, T_low, T_high, yG_low, yG_high)

	x0 	= np.random.rand(Ns,B) 								# initial values of x
	x 	= runTAP(x0, y, Q_process, Q_obs, theta, nltype) 	# run inputs through TAP dynamics

	r = torch.matmul(torch.tensor(U),torch.tensor(x)).data.numpy()
	r += np.random.multivariate_normal(np.zeros(Nr),Q_obs,(B,T)).transpose(0,2,1)

	return y, x, r


def generate_TAPbrain_dynamics(brain, theta, modelparameters, B, T, T_low, T_high, yG_low, yG_high, T_clip, use_cuda):

	"""
	Function that generates the TAP brain dynamics

	Inputs: 
	brain : PyTorch model of the trained TAP brain
	theta : TAP dynamics parameters
	modelparameters 
	B : no. of batches of data
	T : no. of time steps in each batch
	T_low to T_high: range of time period for which the input signal is held constant
	yG_low to y_high: range of input signal gain
	T_clip : no. of time steps to clip to account for burn in period of the TAP brain
	use_cuda


	Outputs:
	y  : input signal           - shape B x Ny x T
	x  : latent dynamics        - shape B x Ns x T
	r  : TAP brain dynamics     - shape B x Nr x T 

	"""

	# generate input data and latent dynamics
	y, x, _ = generate_TAPdynamics(theta, modelparameters, B, T+T_clip, T_low, T_high, yG_low, yG_high)

	# pass the inputs through the brain
	with torch.no_grad():
		r = brain(torch.tensor(y.transpose(0,2,1), dtype=torch.float32))[0]

	# convert to numpy array
	r = r.cpu().data.numpy().transpose(0,2,1)

	# add independent noise to r
	r += np.random.multivariate_normal(np.zeros(modelparameters['Nr']),modelparameters['Q_obs'],[T+T_clip,B]).transpose(1,2,0)
	
	# return clipped arrays
	return y[...,T_clip:], x[...,T_clip:], r[...,T_clip:]



def TAPnonlinearity(x,y,G,J,V,lam):

	"""
	function that computes the TAP nonlinearity in PyTorch
	x_i(t+1) = (1-lam)*x_i(t) + lam*sigmoid(sum_{j,a,b,c} G_{abc}J_{ij}^a x_i(t)^b x_j(t)^c + (Vy(t))_i ) for i=1 to N_s
	
	inputs:
	x : tensor of shape B x Ns x Np (B, Ns, Np = no. of batches, x varibles , particles)
	y : tensor of shape B x Ny x 1  (Ny = no. of input variables)
	G : message passing parameters
	J : interaction matrix, tensor of shape Ns x Ns
	V : input mapping matrix, tensor of shape Ns x Ny
	lam: low pass filtering constant for TAP dynamics 

	output: 
	x(t+1) : tensor of shape B x Ns x Np

	"""

	Ns      = x.shape[1]
	device  = x.device 
	dtype   = x.dtype
	sigmoid = torch.nn.Sigmoid()
	J2      = J**2
	x2      = x**2
	J1      = torch.mm(J,torch.ones((Ns,1),device=device,dtype=dtype)).unsqueeze(0)
	Jx      = torch.matmul(J,x)
	Jx2     = torch.matmul(J,x2)
	J21     = torch.mm(J2,torch.ones((Ns,1),device=device,dtype=dtype)).unsqueeze(0)
	J2x     = torch.matmul(J2,x)
	J2x2    = torch.matmul(J2,x2)

	#argf    = torch.matmul(V,y.unsqueeze(2)) + G[0]*J1 + G[1]*Jx + G[2]*Jx2 + G[9]*J21 + G[10]*J2x + G[11]*J2x2 + x*( G[3]*J1 + G[4]*Jx + G[5]*Jx2 + G[12]*J21 + G[13]*J2x + G[14]*J2x2 ) + x2*(G[6]*J1 + G[7]*Jx + G[8]*Jx2 + G[15]*J21 + G[16]*J2x + G[17]*J2x2)
	argf    = torch.matmul(V,y) + G[0]*J1 + G[1]*Jx + G[2]*Jx2 + G[9]*J21 + G[10]*J2x + G[11]*J2x2 + x*( G[3]*J1 + G[4]*Jx + G[5]*Jx2 + G[12]*J21 + G[13]*J2x + G[14]*J2x2 ) + x2*(G[6]*J1 + G[7]*Jx + G[8]*Jx2 + G[15]*J21 + G[16]*J2x + G[17]*J2x2)
	
	return (1-lam)*x + lam*sigmoid(argf)



def runTAP_torch(x0, y, J, G, V, lam):
    T = y.shape[2]
    xMat = []
    xMat.append(x0)
    xt = x0.unsqueeze(2)
    for t in range(T):
        yt = y[...,t].unsqueeze(2)
        xnew = TAPnonlinearity(xt, yt, G, J, V, lam)
        xMat.append(xnew.squeeze())
        xt = xnew
        
    return torch.stack(xMat).permute(1,2,0)