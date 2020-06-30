import sys
sys.path.append('../code')
from plotutils import *



def main():

	# input parameters
	brain_name      = sys.argv[1]			# brain name
	model_type 		= sys.argv[2]			# model type - 1: Ux + b, 0: Ux
	noise_seed      = int(sys.argv[3])		# set noise seed for repeatability
	Ns      		= int(sys.argv[4])		# assumed no. of latent variables
	B 				= int(sys.argv[5])		# no. of batches of data to use
	Np 				= int(sys.argv[6])		# no. of particles to use
	NIterations     = int(sys.argv[7])		# no. of EM iterations to use
	batch_size 		= int(sys.argv[8])		# batch size for EM
	learning_rate 	= float(sys.argv[9]) 	# learning rate for M step
	alpha_J         = int(sys.argv[10])		# coefficient of L1 penalty on J
	q_obs 			= float(sys.argv[11])	# variance of independent measurement noise to add
	support_G		= int(sys.argv[12]) 	# no. of G coefficients to use

	use_cuda 		= False
	q_process 		= 1e-5


	# set noise seed
	np.random.seed(noise_seed)
	torch.manual_seed(noise_seed)
	print('noise_seed = %d' %(noise_seed))


	# load brain
	print('loading brain ...')
	tapbrain, theta, params = loadbrain('../data/brains/' + brain_name, use_cuda)
	Ns_true 		= params['Ns']
	Ny 				= params['Ny']
	Nr 				= params['Nr']


	# extract the true parameters
	lam, G, J, U, V = extractParams(theta, 18, Ns_true, Ny, Nr)
	baseline 		= params['baseline'] if model_type else 0 		# true baseline activity

	# update observation noise covariances
	Q_obs 				= q_obs*np.eye(Nr)  
	params['Q_obs'] 	= Q_obs


	# generate initial estimates of embedding using ICA
	print('initializing parameters ...')

	B_ICA, T_ICA 	= 2000, 50 
	T_low, T_high 	= 2, 5
	yG_low, yG_high = 50, 50
	T_clip 			= 20
	r_ICA 	= generate_TAPbrain_dynamics(tapbrain,theta, params, B_ICA, T_ICA, T_low, T_high, yG_low, yG_high, T_clip, use_cuda)[2]

	b_init 	= np.mean(r_ICA) if model_type else 0	
	U_init 	= UhatICA(np.reshape(r_ICA.transpose(1,2,0)-b_init, [Nr,B_ICA*T_ICA],order='F').T, Ns)[0]
	del r_ICA

	# initialize the rest of the parameters
	G_init    = np.zeros([18])
	J_init    = 0.1*Create_J(Ns, 0, params['Jtype'], params['self_coupling_on'])  
	if Ns <= Ny:
		V_init = np.linalg.svd(np.random.randn(Ns,Ny), full_matrices=False)[2]
	else:
		V_init = np.linalg.svd(np.random.randn(Ns,Ny), full_matrices=False)[0]



	# generate measurements for particle EM
	print('generating measurements ...')
	T = 25
	yG_low, yG_high = 5, 25
	y, x, r_brain = generate_TAPbrain_dynamics(tapbrain,theta, params, B, T, T_low, T_high, yG_low, yG_high, T_clip, use_cuda)
	print('No of batches of data =',B)


	# inspect and update measurement noise covariance
	SNR, _ , C_err = computeSNR(r_brain-baseline, x, U)
	del x
	print('mean SNR = %.1f' %(SNR))
	print('mean variance of measurement noise = %.3f' %(np.mean(np.diag(C_err))))
	Q_obs = np.mean(np.diag(C_err))*np.eye(Nr)



	# convert data to torch tensors
	device, dtype = "cpu", torch.float64

	Q_pr_true 	= torch.tensor(q_process*np.eye(Ns_true),device=device,dtype=dtype)
	Q_process 	= torch.tensor(q_process*np.eye(Ns),device=device,dtype=dtype)
	Q_obs     	= torch.tensor(Q_obs,device=device,dtype=dtype)

	P_pr_true 	= Q_pr_true.inverse()
	P_process 	= Q_process.inverse()
	P_obs     	= Q_obs.inverse()

	r_brain   	= torch.tensor(r_brain,device=device,dtype=dtype)
	y         	= torch.tensor(y,device=device,dtype=dtype)

	lam       	= torch.tensor(lam,device=device,dtype=dtype,requires_grad=False)

	G 			= torch.tensor(G,device=device,dtype=dtype,requires_grad=False)
	J 			= torch.tensor(J,device=device,dtype=dtype,requires_grad=False)
	U 			= torch.tensor(U,device=device,dtype=dtype,requires_grad=False)
	V 			= torch.tensor(V,device=device,dtype=dtype,requires_grad=False)

	G_hat  		= torch.tensor(G_init,device=device,dtype=dtype,requires_grad=True)
	U_hat  		= torch.tensor(U_init,device=device,dtype=dtype,requires_grad=True)
	V_hat  		= torch.tensor(V_init,device=device,dtype=dtype,requires_grad=True)
	J_hat_vec 	= torch.tensor(JMatToVec(J_init),device=device,dtype=dtype,requires_grad=True)
	J_hat  		= JVecToMat_torch(J_hat_vec,Ns)
	b_hat 		= torch.tensor(b_init,device=device,dtype=dtype,requires_grad=True) if model_type else 0



	# run particle filter with true and initial values of parameters
	print('running particle filter with true and initial parameter values')
	
	B_eval 	= 100 if B>100 else B
 
	with torch.no_grad():
		LL_tp = particlefilter(G, J, U, V, lam, r_brain[0:B_eval]-baseline, y[0:B_eval], P_pr_true, P_obs, Np)[0]


	with torch.no_grad():
		LL_init = particlefilter(G_hat, J_hat, U_hat, V_hat, lam, r_brain[0:B_eval]-b_init, y[0:B_eval], P_process, P_obs, Np)[0]

	print('log likelihood with true parameters = %.1f' %(LL_tp.mean().data.numpy()))
	print('log likelihood with initial parameters = %.1f' %(LL_init.mean().data.numpy()))



	# run particle EM
	T_st 		= 3 # no. of initial time steps to discard in the M-step
	
	if model_type:
	    opt_params = [G_hat, J_hat_vec, U_hat, V_hat, b_hat]
	else:
	    opt_params = [G_hat, J_hat_vec, U_hat, V_hat]

	optimizer  	= torch.optim.Adam(opt_params,lr=learning_rate, betas=(0.9, 0.999))

	LLVec      	= [] 

	print('starting particle EM')

	t_st = time.time()

	for iteration in range(NIterations):

		if iteration == NIterations//2:
			optimizer  = torch.optim.Adam(opt_params,lr=learning_rate/4, betas=(0.9, 0.999))

		if iteration == 3*NIterations//4:
			optimizer  = torch.optim.Adam(opt_params,lr=learning_rate/16, betas=(0.9, 0.999))
			
		#zero-gradients at the start of each iteration
		optimizer.zero_grad() 

		# select indices of batches
		idx = np.random.randint(low=0,high=B,size=batch_size)

		# run particle filter to get posterior for E-step
		with torch.no_grad():
			LL_b, x_b, P_b, W_b = particlefilter(G_hat, JVecToMat_torch(J_hat_vec,Ns), U_hat, V_hat, lam, r_brain[idx]-b_hat, y[idx], P_process, P_obs, Np)

			
		# E-step
		C = Qfunction(G_hat, JVecToMat_torch(J_hat_vec,Ns), U_hat, V_hat, lam, r_brain[idx,:,T_st:]-b_hat, y[idx,:,T_st:], P_b[...,T_st:], W_b, P_process, P_obs)

		# Add L1 norm of JVec
		C += alpha_J*torch.sum(torch.abs(J_hat_vec))

		# M-step
		C.backward() 
		#G_hat.grad[0], G_hat.grad[9] = 0, 0 #set gradient of G0 and G9 to zero
		if support_G < len(G):
			G_hat.grad[support_G:] = 0 
		optimizer.step()

		# record the log likelihood
		LLVec.append(LL_b.mean())

		if (iteration+1) % 500 == 0:
			print('[%d] log likelihood: %.1f' %(iteration + 1, LL_b.mean()))
			
	t_en = time.time()

	print('Finished training')
	print('Time elapsed = %.1f s'%(t_en - t_st))



	# run particle filter with inferred parameters
	J_hat = JVecToMat_torch(J_hat_vec,Ns)
	with torch.no_grad():
		LL_hat = particlefilter(G_hat, J_hat, U_hat, V_hat, lam, r_brain[0:B_eval]-b_hat, y[0:B_eval], P_process, P_obs, Np)[0]

	print('log likelihood with inferred parameters = %.1f' %(LL_hat.mean().data.numpy()))


	# save required data
	print('saving inferred parameters at')
	params['Q_obs'] = Q_obs
	init_parameters = {'G_init':G_init, 'J_init':J_init, 'U_init':U_init, 'V_init':V_init, 'b_init':b_init}
	inferred_parameters = {'G_hat':G_hat.data.numpy(), 'J_hat':J_hat.data.numpy(), 'U_hat':U_hat.data.numpy(), 'V_hat':V_hat.data.numpy(), 'b_hat':b_hat }

	fname = '../data/estimates/' + 'inference_' + brain_name + '.pkl'

	with open(fname, 'wb') as f:  
		pickle.dump([theta, params, init_parameters, inferred_parameters], f)
	f.close()
	
	print(fname)


if __name__ == "__main__":
	main()