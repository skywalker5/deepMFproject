import numpy as np
from sklearn.decomposition import NMF

def compute_obj(X, Ws, Hs):
  m, _    = X.shape
  nlayers = len(Ws)
  Wcomb   = np.eye(m)

  for W in Ws:
    Wcomb = Wcomb.dot(W)

  H       = Hs[-1]

  fit     = 0.5 * np.linalg.norm(X - Wcomb.dot(H))**2

  return fit

def update_AXB(X, A, B, W, options):
  # W <- min_{W >= 0} || X - A W B ||_F^2
  right = B.dot(B.T)
  indep = A.dot(X.dot(B))
  left  = A.dot(A)

  # Lipschitz constant
  L = np.linalg.norm(left, 2) * np.linalg.norm(right, 2)

  # Accerelartion parameters
  alphas = []
  betas  = []
  alphas.append(0.05)
  Y = W.copy()

  itr = 1
  eps0, eps = 0, 1
  errs = []

  while (itr <= options['inneriters']) and (eps >= options['tol']*eps0):
    Wp   = W.copy()
    grad = left.dot(Y.dot(right)) - indep

    alphas.append(0.5 * (np.sqrt(alphas[itr-1]**4 + (4*alphas[itr-1]**2)) - (alphas[itr-1]**2)))
    betas.append(alphas[itr-1]*(1-alphas[itr-1]) / (alphas[itr-1]**2 + alphas[itr]))

    # Projected gradient step from Y
    W = np.maximum((Y - grad) / L, 0)

    # Nesterov step (optimal linear combination of iterates)
    Y = W + betas[itr-1] * (W - Wp)

    fit = 0.5 * np.linalg.norm(X - A.dot(Y.dot(B)))**2
    errs.append(fit)

    # Restart if the error increases
    if (itr >= 2) and (errs[-1] > errs[-2]):
      Y = W.copy()

    if (itr == 1):
      eps = np.linalg.norm(W-Wp)

    eps = np.linalg.norm(W-Wp)
    itr += 1

  return W

def update_layers(X, Ws, Hs, options):
  nlayers = len(Ws)
  m, _    = X.shape

  # Update the Ws
  for l in range(nlayers):
    # Compute the A matrix
    A = np.eye(m)
    for ll in range(l-1):
      A = A.dot(Ws[ll])

    # Compute the B matrix
    _, d = Ws[l].shape
    B    = np.eye(d)
    for rr in range(l, nlayers):
      B = B.dot(Ws[rr])

    B = B.dot(Hs[-1])


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

  # Initialise the layers (sequential algo)
  if (options['verbose']):
    print('Starting Deep MF')
    print('Initialise all the layers via NMF')

  Ws   = []
  Hs   = []
  errs = []
  Xfac = X.copy()

  for l in range(nlayers):
    if (options['verbose']):
      print(f'Initialising layer {l}')

    # Approximate H_l as W_{l+1}H_{l+1}
    model = NMF(inner_ranks[l])
    Ws[l] = model.fit_transform(Xfac)
    Hs[l] = model.components_
    Xfac  = Hs[l].copy()

  # Compute initial error
  errs.append(compute_obj(X, Ws, Hs))

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
      Ws, Hs = update_layers(X, Ws, Hs, options)

      errs.append(compute_obj(X, Ws, Hs))
      if (options['verbose']):
        print(f"Iteration {itr} Fit {errs[-1]:.4f}")

      if (itr > options['maxiter']) or (errs[-1] > errs[-2] * (1 - options['tol'])):
        stop = True

  return Ws, Hs 
