import warnings
warnings.filterwarnings('ignore')

from utils import *
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


def computeSNR(r_brain, x, U):
	
	"""
	function for computing the SNR of the TAP brain measurements
	"""
	
	B, Nr, T    = r_brain.shape
	r_sig       = r_brain*0

	for b in range(B):
		r_sig[b] = np.dot(U,x[b])

	dr = r_brain - r_sig

	dr = dr.transpose(1,2,0)          # Nr x T x B

	r_sig = r_sig.transpose(1,2,0)    # Nr x T x B

	# subsample and compute covariances
	r_sig = r_sig[:,::3]
	C_sig = np.cov(np.reshape(r_sig,[Nr,B*r_sig.shape[1]]))

	dr    = dr[:,::3]
	C_err = np.cov(np.reshape(dr,[Nr,B*dr.shape[1]]))

	SNR   = np.mean(np.diag(C_sig)/np.diag(C_err))

	return SNR, C_sig, C_err


def createModelParameters(Ns, Nr, Ny, block_diagonalJ, sparsity_J, self_coupling_on, Jtype, gain_J, model_type):
	G   = np.array([0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]) # message passing parameters of the TAP equation

	block_diagonalJ = 0
	if block_diagonalJ:
		J = np.zeros([Ns,Ns])
		M = Ns//4
		J[0:M,0:M] = 1*Create_J(M, sparsity_J, 'ferr', self_coupling_on)
		J[M:2*M,M:2*M] = 1*Create_J(M, sparsity_J, 'antiferr', self_coupling_on)
		J[2*M:,2*M:] = gain_J*Create_J(Ns-2*M, 0.25, 'nonferr', self_coupling_on)
		del M
	else:
		self_coupling_on, sparsity_J, gain_J, Jtype  = 1, 0, 3, Jtype # interaction matrix settings
		J = gain_J*Create_J(Ns, sparsity_J, Jtype, self_coupling_on) # interaction matrix 

	if model_type:
		gain_U = 1
		U   = gain_U*np.random.randn(Nr,Ns) # embedding matrix
	else:
		gain_U = 3
		U   = gain_U*np.random.rand(Nr,Ns) # embedding matrix

	if Ns <= Ny:
		V = np.linalg.svd(np.random.randn(Ns,Ny), full_matrices=False)[2]
	else:
		V = np.linalg.svd(np.random.randn(Ns,Ny), full_matrices=False)[0]

	return G, J, U, V



def permuteJ(J, PermMat):
	Ns = J.shape[0]
	Jhat = J*0
	x_idx = np.dot(PermMat.T, np.arange(Ns))

	for ii in range(Ns):
		for jj in range(Ns):
			Jhat[ii,jj] = J[np.int(x_idx[ii]),np.int(x_idx[jj])]
	return Jhat


def affine_transform(G_hat, J_true_vec, J_hat_vec):
	z = np.polyfit(J_true_vec, J_hat_vec,1)
	m, b = z[0], z[1]
	b = 0
	# normalize G_hat and J_hat
	J_hat_vec_normalized = (J_hat_vec - b)/m

	G_hat_normalized = G_hat*1.0
	G_hat_normalized[0:9] = m*G_hat[0:9] + 2*m*b*G_hat[9:]
	G_hat_normalized[9:] = (m**2)*G_hat[9:]
	return G_hat_normalized, J_hat_vec_normalized