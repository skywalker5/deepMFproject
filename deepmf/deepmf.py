import numpy as np
from sklearn.decomposition import NMF
from nnls import nnlsm_blockpivot

def compute_obj(X, Ws, H):
  m, _    = X.shape
  nlayers = len(Ws)
  Wcomb   = np.eye(m)

  for W in Ws:
    Wcomb = Wcomb.dot(W)

  fit     = 0.5 * np.linalg.norm(X - Wcomb.dot(H))**2

  return fit

def update_AX(X, A, W, options):
  # W <- min_{W >= 0} || X - A W ||_F^2
  indep = A.T.dot(X)
  left  = A.T.dot(A)

  # Lipschitz constant
  L = np.linalg.norm(left, 2)

  # Accerelartion parameters
  alphas = []
  betas  = []
  alphas.append(0.05)

  # Ensure initialisation is in feasible set
  W = np.maximum(W, 0)
  Y = W.copy()

  itr = 1
  eps0, eps = 0, 1
  errs = []

  while (itr <= options['inneriters']) and (eps >= options['tol']*eps0):
    # Previous iterate
    Wp   = W.copy()
    grad = left.dot(Y) - indep

    alpha = 0.5 * (np.sqrt(alphas[itr-1]**4 + (4*alphas[itr-1]**2)) - (alphas[itr-1]**2))
    alphas.append(alpha)    

    beta = alphas[itr-1] * (1-alphas[itr-1]) / ((alphas[itr-1]**2 + alphas[itr]))
    betas.append(beta)

    # Projected gradient step from Y
    W = np.maximum(Y - grad / L, 0)

    # Nesterov step (optimal linear combination of iterates)
    Y = W + betas[itr-1] * (W - Wp)

    fit = 0.5 * np.linalg.norm(X - A.dot(W))**2
    errs.append(fit)

    if (options['verbose']):
      print(f'Inner iteration {itr} fit {fit:.4f}')

    # Restart if the error increases
    if (itr >= 2) and (errs[-1] > errs[-2]):
      Y = W.copy()

    if (itr == 1):
      eps0 = np.linalg.norm(W-Wp)

    eps = np.linalg.norm(W-Wp)
    itr += 1

  return W

def update_AXB(X, A, B, W, options):
  # W <- min_{W >= 0} || X - A W B ||_F^2
  right = B.dot(B.T)
  indep = A.T.dot(X.dot(B.T))
  left  = A.T.dot(A)

  # Lipschitz constant
  L = np.linalg.norm(left, 2) * np.linalg.norm(right, 2)

  # Accerelartion parameters
  alphas = []
  betas  = []
  alphas.append(0.05)

  # Ensure initialisation is in feasible set
  W = np.maximum(W, 0)
  Y = W.copy()

  itr = 1
  eps0, eps = 0, 1
  errs = []

  while (itr <= options['inneriters']) and (eps >= options['tol']*eps0):
    # Previous iterate
    Wp   = W.copy()
    grad = left.dot(Y.dot(right)) - indep

    alpha = 0.5 * (np.sqrt(alphas[itr-1]**4 + (4*alphas[itr-1]**2)) - (alphas[itr-1]**2))
    alphas.append(alpha)    

    beta = alphas[itr-1] * (1-alphas[itr-1]) / ((alphas[itr-1]**2 + alphas[itr]))
    betas.append(beta)

    # Projected gradient step from Y
    W = np.maximum(Y - grad / L, 0)

    # Nesterov step (optimal linear combination of iterates)
    Y = W + betas[itr-1] * (W - Wp)

    fit = 0.5 * np.linalg.norm(X - A.dot(W.dot(B)))**2
    errs.append(fit)

    if (options['verbose']):
      print(f'Inner iteration {itr} fit {fit:.4f}')

    # Restart if the error increases
    if (itr >= 2) and (errs[-1] > errs[-2]):
      Y = W.copy()

    if (itr == 1):
      eps0 = np.linalg.norm(W-Wp)

    eps = np.linalg.norm(W-Wp)
    itr += 1

  return W

def update_layers(X, Ws, H, options):
  nlayers = len(Ws)
  m, _    = X.shape
  
  # Update the first W
  if (options['verbose']):
    print("Updating the W factors")

  A = np.eye(Ws[0].shape[1])
  for l in range(1, nlayers):
    A = A.dot(Ws[l])
  A = A.dot(H)

  if (options['nnlsm']):
    if (options['verbose']):
      print(f"Updating W[0] via BPP")
    AAt      = A.dot(A.T)
    ABt      = A.dot(X.T)
    Wt, flag = nnlsm_blockpivot(AAt, ABt, is_input_prod=True, init=Ws[0].T)
  else:
    if (options['verbose']):
      print(f"Updating W[0] via gradient descent")
    Wt = update_AX(X.T, A.T, Ws[0].T, options)
    
  Ws[0]    = Wt.T

  # Update the inner Ws
  for l in range(1, nlayers):
    if (options['verbose']):
      print(f"Updating W[{l}]")

    # Compute the A matrix
    A = np.eye(m)
    for ll in range(l):
      A = A.dot(Ws[ll])

    # Compute the B matrix
    _, d = Ws[l].shape
    B    = np.eye(d)
    for rr in range(l+1, nlayers):
      B = B.dot(Ws[rr])

    B = B.dot(H)

    Ws[l] = update_AXB(X, A, B, Ws[l], options)

  # Update H
  A = np.eye(m)
  for l in range(nlayers):
    A = A.dot(Ws[l])
 
  if (options['nnlsm']): 
    if (options['verbose']):
      print("Updating H via BPP")
    AtA     = A.T.dot(A)
    AtB     = A.T.dot(X)
    H, flag = nnlsm_blockpivot(AtA, AtB, is_input_prod=True, init=H)
  else:
    if (options['verbose']):
      print("Updating H via gradient descent")
    H = update_AX(X, A, H, options)

  return Ws, H

def deepmf(X, nlayers, inner_ranks, options):
  if (not 'sequential' in options):
    options['sequential'] = False

  if (not 'inneriters' in options):
    options['inneriters'] = 50

  if (not 'tol' in options):
    options['tol'] = 1e-6

  if (not 'maxiter' in options):
    options['maxiter'] = 100

  if (not 'verbose' in options):
    options['verbose'] = False

  if (not 'nnlsm' in options):
    options['nnlsm'] = False

  # Initialise the layers (sequential algo)
  if (options['verbose']):
    print('Starting Deep MF')
    print('Initialise all the layers via NMF')

  Ws   = []
  errs = []
  Xfac = X.copy()

  for l in range(nlayers):
    if (options['verbose']):
      print(f'Initialising layer {l} with rank {inner_ranks[l]}')

    # Approximate H_l as W_{l+1}H_{l+1}
    model = NMF(inner_ranks[l])
    W     = model.fit_transform(Xfac)
    Ws.append(W)
    Xfac  = model.components_

  # Compute initial error
  H = Xfac.copy()
  errs.append(compute_obj(X, Ws, H))

  # Iterate if necessary
  if (not options['sequential']):
    if (options['verbose']):
      print('Starting the iterative updates')

    stop = False
    itr  = 0
    if (options['verbose']):
      print(f"Iteration {itr} Fit {errs[-1]:.4f}")
  
    while (not stop):
      itr += 1
      if (options["verbose"]):
        print(f"Starting iteration {itr}")

      # Update the factor matrices
      Ws, H = update_layers(X, Ws, H, options)

      errs.append(compute_obj(X, Ws, H))
      if (options['verbose']):
        print(f"Iteration {itr} Fit {errs[-1]:.4f}")

      if (itr > options['maxiter']) or (errs[-1] > errs[-2] * (1 - options['tol'])):
        stop = True

  return Ws, H, errs 
