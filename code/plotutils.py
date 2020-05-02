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