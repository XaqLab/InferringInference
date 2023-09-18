import numpy as np
from utils import *
from tapdynamics import TAPnonlinearity
import torch
import torch.distributions.multivariate_normal as MVN


def particlefilter(G, J, U, V, lam, r, y, P_process, P_obs, Np):

    """
    particle filter function for estimating latent dynamics of the TAP brain

    B, Ns, Nr, Ny, T = no. of batches, latent varibles, neurons, input variables, time steps

    inputs:
    G : message passing parameters
    J : interaction matrix, tensor of shape Ns x Ns
    U : embedding matrix. tensor of shape Nr x Ns
    V : input mapping matrix, tensor of shape Ns x Ny
    lam: low pass filtering constant for TAP dynamics
    r : tensor of shape B x Nr x T
    y : tensor of shape B x Ny      (Ny = no. of input variables)
    P_process : inverse of covariance of process noise, tensor of shape Ns x Ns
    P_obs : inverse of covariance of observation noise, tensor of shape Nr x Nr
    Np : no. of particles to use

    outputs:
    LL  : observed data log likelihood, tensor of shape B
    xhat: estimated latent dynamics, tensor of shape B x Ns x T
    ParticlesAll : particle trajectories, tensor of shape B x Ns x Np x T
    WVec:particle weights, tensor of shape B x Np

    """

    if len(r.shape) < 3:
        r.unsqueeze_(0)  # this is to ensure shape is B x Nr x T
        y.unsqueeze_(0)  # this is to ensure shape is B x Ny x T

    B, Nr, T = r.shape  # no. of batches, no. of neurons, no. of time steps
    Ns, Ny = J.shape[0], y.shape[0]  # no. of latent variables, no. of inputs
    device = r.device
    dtype = r.dtype

    # Compute the inverse covariance of the proposal distribution q = p(x_t | x_(t-1), r_t)
    # Notation: Q for covariance, P for inverse covariance
    P_1 = torch.mm(P_obs, U)  # intermediate matrix 1
    P_2 = torch.mm(U.t(), P_1)  # intermediate matrix 2
    P_proposal = P_process + P_2
    P_proposal = (
        P_proposal + P_proposal.t()
    ) / 2  # make it a perfectly symmetric matrix
    Q_proposal = P_proposal.inverse()

    # Define the noise processes
    mvn_process = MVN.MultivariateNormal(
        loc=torch.zeros(Ns, device=device, dtype=dtype), precision_matrix=P_process
    )
    mvn_proposal = MVN.MultivariateNormal(
        loc=torch.zeros(Ns, device=device, dtype=dtype), precision_matrix=P_proposal
    )

    # Generate initial particles X_0 ~ N( pinv(U)r_0, Q_process )
    # At the time step zero generate K x Np particles and pick Np particles with highest weights
    K = 10
    r0 = r[..., 0]
    x = torch.matmul(torch.pinverse(U), r0.unsqueeze(2)) + mvn_process.rsample(
        sample_shape=torch.Size([B, K * Np])
    ).permute(0, 2, 1)

    # Compute the initial weights of the particles
    P_3 = torch.mm(torch.pinverse(U).t(), P_process)  #  intermediate matrix 3
    logWVec = -0.5 * (
        (x * torch.matmul(P_2, x)).sum(dim=1)
        - 2 * (r0.unsqueeze(2) * torch.matmul(P_1, x)).sum(dim=1)
    )
    logWVec += 0.5 * (
        (x * torch.matmul(P_process, x)).sum(dim=1)
        - 2 * (r0.unsqueeze(2) * torch.matmul(P_3, x)).sum(dim=1)
    )
    log_e = torch.max(logWVec, dim=1)[0]  # find maximum log weight
    logWVec -= log_e.unsqueeze(1)  # subtract the maximum

    # retain only the Np best particles
    for b in range(B):
        idx = torch.argsort(logWVec[b], descending=True)
        logWVec[b] = logWVec[b, idx]
        x[b] = x[b, :, idx]

    logWVec, x = logWVec[:, 0:Np], x[..., 0:Np]

    # normalized initial weights
    WVec = torch.exp(logWVec)
    WVec = WVec / torch.sum(WVec, dim=1).unsqueeze(1)

    ParticlesAll = torch.zeros((B, Ns, Np, T), device=device, dtype=dtype)
    ParticlesAll[..., 0] = x

    # normalization constant for the weights
    # log_nu = 0.5*np.log(np.linalg.det(P_process.data.numpy())) +  0.5*np.log(np.linalg.det(P_obs.data.numpy())) -0.5*np.log(np.linalg.det(P_proposal.data.numpy()))
    # log_nu += -0.5*Nr*np.log(2*np.pi)
    log_nu = 0

    # # Compute log p(r_0) to initialize the observed data log likelihood
    # P_4 = torch.mm(P_1, torch.mm(P_2.inverse(),P_1.t()))
    # LL = -0.5*(Nr - Ns)*np.log(2*np.pi) + 0.5*np.log(np.linalg.det(P_obs.data.numpy())) - 0.5*np.log(np.linalg.det(P_2.data.numpy()))
    # if B == 1:
    #     LL += -0.5*torch.matmul(r0.unsqueeze(1),torch.matmul(P_obs - P_4,r0.unsqueeze(2))).item()
    # else:
    #     LL += -0.5*torch.matmul(r0.unsqueeze(1),torch.matmul(P_obs - P_4,r0.unsqueeze(2))).squeeze()
    LL = 0

    # savedparticles = []
    # savedparticles.append(x[0].data.numpy())

    for tt in range(1, T):

        # resample particles based on their weights if sample diversity is low
        ESS = 1 / torch.sum(WVec ** 2, dim=1)

        for b in range(B):
            if ESS[b] < Np / 2 and tt != T - 1:
                idx = resampleSystematic_torch(WVec[b], Np, device, dtype)
                ParticlesAll[b] = ParticlesAll[b, :, idx]

        x = ParticlesAll[..., tt - 1]

        yt = y[..., tt - 1]
        rt = r[..., tt]

        Minvr = torch.matmul(P_obs, rt.unsqueeze(2))  # size B x Nr x 1
        rMinvr = torch.matmul(rt.unsqueeze(1), Minvr)  # size B x 1  x 1
        UMinvr = torch.matmul(U.t(), Minvr)  # size B x Ns x 1

        f_tap = TAPnonlinearity(x, yt.unsqueeze(2), G, J, V, lam)

        Pinvf_tap = torch.matmul(P_process, f_tap)
        v = Pinvf_tap + UMinvr
        mu_proposal = torch.matmul(Q_proposal, v)  # mean of the proposal distribution

        # sample new particles from proposal distribution
        ParticlesNew = mu_proposal + mvn_proposal.rsample(
            sample_shape=torch.Size([B, Np])
        ).permute(0, 2, 1)

        # log of incremental weights
        log_alpha = log_nu - 0.5 * (
            rMinvr.squeeze(2) + torch.sum(f_tap * Pinvf_tap - v * mu_proposal, dim=1)
        )

        # update log weights
        logWVec = torch.log(WVec) + log_alpha
        log_e = torch.max(logWVec, dim=1)[0]  # find maximum log weight
        logWVec -= log_e.unsqueeze(1)  # subtract the maximum

        # unnormalized weights
        WVec = torch.exp(logWVec)

        # update log likelihood
        LL += torch.log(torch.sum(WVec, dim=1)) + log_e

        # normalize the weights
        WVec = WVec / torch.sum(WVec, dim=1).unsqueeze(1)

        # append particles
        ParticlesAll[..., tt] = ParticlesNew

        # savedparticles.append(np.copy(ParticlesAll[0,:,:,0:tt+1].data.numpy()))
        # savedparticles.append(ParticlesNew[0].data.numpy())

    xhat = torch.sum(ParticlesAll * WVec.view(B, 1, Np, 1), dim=2).squeeze(2)

    return LL, xhat, ParticlesAll, WVec


def Qfunction(G, J, U, V, lam, r, y, Particles, Weights, P_process, P_obs):

    """
    Q function computed in the E step of the EM algorithm
    A particle approximation of the required posterior distribution is used

    B, Ns, Nr, Ny, T = no. of batches, latent varibles, neurons, input variables, time steps

    inputs:
    G : message passing parameters
    J : interaction matrix, tensor of shape Ns x Ns
    U : embedding matrix. tensor of shape Nr x Ns
    V : input mapping matrix, tensor of shape Ns x Ny
    lam: low pass filtering constant for TAP dynamics
    r : tensor of shape B x Nr x T
    y : tensor of shape B x Ny      (Ny = no. of input variables)
    Particles : particle trajectories, tensor of shape B x Ns x Np x T
    Weights   : particle weights, tensor of shape B x Np
    P_process : inverse of covariance of process noise, tensor of shape Ns x Ns
    P_obs     : inverse of covariance of observation noise, tensor of shape Nr x Nr

    outputs:
    C : Sum of the Q function over all batches of data
    """

    T = r.shape[-1]  # no. of time steps

    # two components of the cost
    C1, C2 = 0, 0

    for t in range(1, T):

        rt = r[..., t]
        yt = y[..., t - 1]
        x = Particles[..., t - 1]
        x_curr = Particles[..., t]

        x_pred = TAPnonlinearity(x, yt.unsqueeze(2), G, J, V, lam)

        dx = x_curr - x_pred
        dr = rt.unsqueeze(2) - torch.matmul(U, x_curr)

        # update the cost
        C1 += 0.5 * torch.sum(dx * torch.matmul(P_process, dx) * Weights.unsqueeze(1))
        C2 += 0.5 * torch.sum(dr * torch.matmul(P_obs, dr) * Weights.unsqueeze(1))

    # Add the L1 norms of G and J
    C = C1 + C2

    return C
