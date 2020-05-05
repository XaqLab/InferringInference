import sys
sys.path.append('../code')
from rnnmodel import *
from plotutils import *


def generateModelParameters(Ns, Nr, Ny, block_diagonalJ, model_type):
	"""
	Parameters and settings used to generate TAP dynamics
	"""
	# process and observation noise covariance matrices
	q_process, q_obs = 0, 0
	Q_process, Q_obs = q_process*np.eye(Ns), q_obs*np.eye(Nr)

	# filter used for smoothing the input signals
	smoothing_filter = signal.hamming(5,sym=True) 
	smoothing_filter = smoothing_filter/sum(smoothing_filter)


	# ground truth TAP model parameters

	lam = np.array([0.25])  # low pass filtering constant for the TAP dynamics

	G   = np.array([0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]) # message passing parameters of the TAP equation

	if block_diagonalJ:
		self_coupling_on, sparsity_J, gain_J, Jtype  = 1, 0, 3, 'nonferr'
		J = np.zeros([Ns,Ns])
		M = Ns//4
		J[0:M,0:M] = 1*Create_J(M, sparsity_J, 'ferr', self_coupling_on)
		J[M:2*M,M:2*M] = 1*Create_J(M, sparsity_J, 'antiferr', self_coupling_on)
		J[2*M:,2*M:] = gain_J*Create_J(Ns-2*M, 0.25, 'nonferr', self_coupling_on)
	else:
		self_coupling_on, sparsity_J, gain_J, Jtype  = 1, 0.25, 3, 'nonferr' # interaction matrix settings
		J 	= gain_J*Create_J(Ns, sparsity_J, Jtype, self_coupling_on) # interaction matrix 

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

	# concatenate parameters
	theta = np.concatenate([lam, G, JMatToVec(J), U.flatten('F'), V.flatten('F') ])

	params = {'Ns':Ns,'Ny':Ny,'Nr':Nr,'Q_process':Q_process,'Q_obs':Q_obs,'nltype':'sigmoid','smoothing_filter':smoothing_filter,'self_coupling_on':self_coupling_on,'sparsity_J':sparsity_J,'Jtype':Jtype }


	return theta, params


def createbrain(N_input, N_hidden, N_output, use_cuda):
	"""
	Create RNN module
	"""
	brain = RNN(N_input, N_hidden, N_output, use_cuda)

	if use_cuda and torch.cuda.is_available():
		brain.cuda()

	return brain


def trainbrain(brain, y_train, y_val, r_train, r_val, NEpochs, batch_size, T_clip, learning_rate, use_cuda):

	loss_fn   	= nn.MSELoss()
	optimizer 	= optim.Adam(brain.parameters(), lr=learning_rate, betas=(0.9, 0.999))

	B_train 	= y_train.shape[0]
	epoch 		= B_train//batch_size 	# no. of iterations in one epoch
	NIterations = epoch*NEpochs 		# total no. of iterations

	# convert training data to torch tensors
	y_train = torch.tensor(y_train.transpose(0,2,1), dtype=torch.float32) # input signal
	r_train = torch.tensor(r_train.transpose(0,2,1), dtype=torch.float32) # target neural activity

	y_val = torch.tensor(y_val.transpose(0,2,1), dtype=torch.float32) # input signal
	r_val = torch.tensor(r_val.transpose(0,2,1), dtype=torch.float32) # target neural activity

	if use_cuda and torch.cuda.is_available():
		y_train = y_train.cuda()
		r_train = r_train.cuda()
		y_val 	= y_val.cuda()
		r_val 	= r_val.cuda()


	train_loss, val_loss = [], []

	t_st = time.time()

	

	for iteration in range(NIterations):

		if iteration == NIterations//2:
			optimizer = optim.Adam(brain.parameters(), lr=learning_rate/2, betas=(0.9, 0.999))

		if iteration == 3*NIterations//4:
			optimizer = optim.Adam(brain.parameters(), lr=learning_rate/4, betas=(0.9, 0.999))

		
		optimizer.zero_grad() # zero-gradients at the start of each iteration
		
		# training data
		
		batch_index = torch.randint(0, B_train,(batch_size,))

		rhat_train  = brain(y_train[batch_index])[0]
		
		mse_train = loss_fn(r_train[batch_index,T_clip:], rhat_train[:,T_clip:])
		
		mse_train.backward()  
		
		optimizer.step() 

		if (iteration + 1) % epoch == 0:
			with torch.no_grad(): 
				rhat_val = brain(y_val)[0]
				mse_val  = loss_fn(r_val[:,T_clip:], rhat_val[:,T_clip:])
			train_loss.append(mse_train.item())
			val_loss.append(mse_val.item())
		
		if (iteration + 1) % 5000 == 0:
			print('[%d] training loss: %.5f' %(iteration + 1, mse_train.item()))

	print('Finished training')
	t_en = time.time()
	print('Time elapsed = %.2f s' %(t_en - t_st))
	
	brain.cpu()
	brain.use_cuda = False
		
	return brain, train_loss, val_loss


def main():

	# Input parameters
	noise_seed      = int(sys.argv[1]) 
	Ns              = int(sys.argv[2]) 	# No. of latent variables
	Nr              = int(sys.argv[3]) 	# No. of neurons
	Ny              = int(sys.argv[4]) 	# No. of input variables
	N_hidden        = int(sys.argv[5]) 	# No. of hidden units in the RNN
	B_train 		= int(sys.argv[6]) 	# No. of training data batches
	NEpochs     	= int(sys.argv[7]) 	# No. of training epochs
	batch_size 		= int(sys.argv[8]) 	# batch size for training
	learning_rate   = float(sys.argv[9])
	block_diagonalJ = int(sys.argv[10]) # 1 for block diagonal J matrix
	model_type 		= int(sys.argv[11]) # training data model - 1: Ux + b, 0: Ux	

	B_val 			= 500      			# No. of batches in validation data
	T 				= 50 				# No. of time steps in each batch 
	T_clip  		= 20              	# No. of time steps to clip
	T_low, T_high 	= 2, 5             	# range of time periods for which input is held constant
	yG_low, yG_high = 5, 50          	# range of input gains
	use_cuda 		= True 				# use cuda if available

	# set noise seed
	np.random.seed(noise_seed)
	torch.manual_seed(noise_seed)
	print('noise_seed = %d' %(noise_seed))

	# generate model parameters
	theta, params   = generateModelParameters(Ns, Nr, Ny, block_diagonalJ, model_type)

	# generate training data
	print('Generating training data ...')
	y_train, _, r_train = generate_TAPdynamics(theta, params, B_train, T+T_clip, T_low, T_high, yG_low, yG_high)
	y_val, _, r_val 	= generate_TAPdynamics(theta, params, B_val, T+T_clip, T_low, T_high, yG_low, yG_high)

	# add basline activity to the targets
	baseline = -np.min(r_train) if model_type else 0
	r_train += baseline
	r_val 	+= baseline


	# create the tap brain 
	tapbrain = createbrain(Ny, N_hidden, Nr, use_cuda)

	# train the tap brain
	print('Training brain ... ')
	tapbrain, train_loss, val_loss = trainbrain(tapbrain, y_train, y_val, r_train, r_val, NEpochs, batch_size, T_clip, learning_rate, use_cuda)
	print('Training complete. Saving brain at')

	# save the tap brain
	brain_name = '../data/brains/' + 'Ns_'+ str(Ns) + '_noiseseed_' + str(noise_seed)
	torch.save(tapbrain, brain_name + '.pt')
	print(brain_name + '.pt')

	# save the brain parameters
	params['train_loss'] = train_loss
	params['val_loss'] 	 = val_loss
	params['baseline'] 	 = baseline

	with open(brain_name + '_params.pkl', 'wb') as f:  
		pickle.dump([theta, params], f)
	f.close()



if __name__ == "__main__":
	main()