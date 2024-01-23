#!/usr/bin/env python3
# coding: utf-8

# ## Optimize acquisition functions using torch.optim
# 
# In this tutorial, we show how to use PyTorch's `optim` module for optimizing BoTorch MC acquisition functions. This is useful if the acquisition function is stochastic in nature (caused by re-sampling the base samples when using the reparameterization trick, or if the model posterior itself is stochastic).
# 
# *Note:* A pre-packaged, more user-friendly version of the optimization loop we will develop below is contained in the `gen_candidates_torch` function in the `botorch.gen` module. This tutorial should be quite useful if you would like to implement custom optimizers beyond what is contained in `gen_candidates_torch`.
# 
# As discussed in the [CMA-ES tutorial](./optimize_with_cmaes), for deterministic acquisition functions BoTorch uses quasi-second order methods (such as L-BFGS-B or SLSQP) by default, which provide superior convergence speed in this situation. 

# ### Set up a toy model
# 
# We'll fit a `SingleTaskGP` model on noisy observations of the function $f(x) = 1 - \|x\|_2$ in `d=5` dimensions on the hypercube $[-1, 1]^d$.

# In[1]:


import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import pdb


# In[2]:


d = 5

bounds = torch.stack([-torch.ones(d), torch.ones(d)])

train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(50, d)
train_Y = 1 - torch.norm(train_X, dim=-1, keepdim=True)

model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll);


# ### Define acquisition function
# 
# We'll use `qExpectedImprovement` with a custom sampler that uses a small number of MC samples and re-samples upon each evaluation of the function. This results in a stochastic acquisition function that one should not attempt to optimize with the quasi-second order methods that are used by default in BoTorch's `optimize_acqf` function.

# In[3]:


from botorch.acquisition import qExpectedImprovement
from botorch.sampling import IIDNormalSampler

sampler = IIDNormalSampler(num_samples=100, resample=True)
qEI = qExpectedImprovement(model, best_f=train_Y.max(), sampler=sampler)


# ### Optimizing the acquisition function
# 
# We will perform optimization over `N=5` random initial `q`-batches with `q=2` in parallel. We use `N` random restarts because the acquisition function is non-convex and as a result we may get stuck in local minima.

# In[4]:


N = 5
q = 2


# #### Choosing initial conditions via a heuristic
# 
# Using random initial conditions in conjunction with gradient-based optimizers can be problematic because qEI values and their corresponding gradients are often zero in large parts of the feature space. To mitigate this issue, BoTorch provides a heuristic for generating promising initial conditions (this dirty and not-so-little secret of Bayesian Optimization is actually very important for overall closed-loop performance).
# 
# Given a set of `q`-batches $X'$ and associated acquisiton function values $Y'$, the `initialize_q_batch_nonneg` samples promising initial conditions $X$ (without replacement) from the multinomial distribution
# 
# $$ \mathbb{P}(X = X'_i) \sim \exp (\eta \tilde{Y}_i), \qquad \text{where} \;\; \tilde{Y}_i = \frac{Y'_i - \mu(Y)}{\sigma(Y)} \;\; \text{if} \;\; Y'_i >0 $$
# 
# and $\mathbb{P}(X = X'_j) = 0$ for all $j$ such that $Y'_j = 0$. 
# 
# Fortunately, thanks to the high degree of parallelism in BoTorch, evaluating the acquisition function at a large number of randomly chosen points is quite cheap.

# In[5]:


from botorch.optim.initializers import initialize_q_batch_nonneg

# generate a large number of random q-batches
Xraw = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(100 * N, q, d)
Yraw = qEI(Xraw)  # evaluate the acquisition function on these q-batches

# apply the heuristic for sampling promising initial conditions
X = initialize_q_batch_nonneg(Xraw, Yraw, N)

# we'll want gradients for the input
X.requires_grad_(True);


# #### Optimizing the acquisition function
# 
# If you have used PyTorch, the basic optimization loop should be quite familiar. However, it is important to note that there is a **key difference** here compared to training ML models: When training ML models, one typically computes the gradient of an empirical loss function w.r.t. the model's parameters, while here we take the gradient of the acquisition function w.r.t. to the candidate set.
# 
# Thus, when setting the optimizer from `torch.optim`, we **do not** add the acquisition function's parameters as parameters to optimize (that would be quite bad!).
# 
# In this example, we use a vanilla `Adam` optimizer with fixed learning rate for a fixed number of iterations in order to keep things simple. But you can get as fancy as you want with learning rate scheduling, early termination, etc.
# 
# A couple of things to note:
# 1. Evaluating the acquisition function on the `N x q x d`-dim inputs means evaluating `N` `q`-batches in `t`-batch mode. The result of this is an `N`-dim tensor of acquisition function values, evaluated independently. To compute the gradient of the full input `X` via back-propagation, we can for convenience just compute the gradient of the sum of the losses. 
# 2. `torch.optim` does not have good built in support for constraints (general constrained stochastic optimization is hard and still an open research area). Here we do something simple and project the value obtained after taking the gradient step to the feasible set - that is, we perform "projected stochastic gradient descent". Since the feasible set here is a hyperrectangle, this can be done by simple clamping. Another approach would be to transform the feasible interval for each dimension to the real line, e.g. by using a sigmoid function, and then optimizing in the unbounded transformed space. 

# In[6]:


# set up the optimizer, make sure to only pass in the candidate set here
optimizer = torch.optim.Adam([{"params": X}], lr=0.01)
X_traj = []  # we'll store the results

# run a basic optimization loop
for i in range(75):
    optimizer.zero_grad()
    # this performs batch evaluation, so this is an N-dim tensor
    pdb.set_trace()
    losses = - qEI(X)  # torch.optim minimizes
    loss = losses.sum()
    
    loss.backward()  # perform backward pass
    #pdb.set_trace()
    optimizer.step()  # take a step
    
    # clamp values to the feasible set
    for j, (lb, ub) in enumerate(zip(*bounds)):
        X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
    
    # store the optimization trajecatory
    X_traj.append(X.detach().clone())
    
    if (i + 1) % 15 == 0:
        print(f"Iteration {i+1:>3}/75 - Loss: {loss.item():>4.3f}")
    
    # use your favorite convergence criterion here...


# In[ ]:




