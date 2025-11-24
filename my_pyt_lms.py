import numpy as np


# def my_pyt_lms(x, d, nSteps, N, w=None, lock_coeff=False):
#     """
#     LMS (Least Mean Squares) adaptive filter for signal prediction.
    
#     This function implements the LMS algorithm, which is an adaptive filtering
#     technique that iteratively adjusts filter weights to minimize the mean square
#     error between the desired signal and the predicted output. The algorithm uses
#     gradient descent to update weights based on the instantaneous error.
    
#     Algorithm Steps:
#     1. Initialize weights to zeros if not provided
#     2. For each time sample (row) in the input signal:
#        - Flip the input row to create the input vector
#        - Compute prediction using current weights
#        - Calculate the error between desired and predicted signals
#        - Update weights using LMS gradient descent (if not locked)
#        - Store weight history
#     3. Return predictions, errors, and weight evolution
    
#     Parameters
#     ----------
#     x : numpy.ndarray
#         Input signal matrix where each row represents a time sample.
#         Shape: (L, N) where L is the number of samples and N is the filter order.
#     d : numpy.ndarray
#         Desired signal vector (target values). Shape: (L,)
#     nSteps : int
#         Latency parameter. Accepted for API compatibility but not used in the
#         main algorithm loop.
#     N : int
#         Filter order (number of taps/weights in the adaptive filter).
#     w : numpy.ndarray, optional
#         Initial weight vector of shape (N,). If None, weights are initialized
#         to zeros. Default is None.
#     lock_coeff : bool, optional
#         Boolean flag to freeze weight updates. When True, weights remain constant
#         throughout the filtering process (useful for testing with pre-trained
#         weights). Default is False.
    
#     Returns
#     -------
#     tuple of (y, e, wts)
#         y : numpy.ndarray
#             Prediction vector with the same length as d. Shape: (L,)
#         e : numpy.ndarray
#             Error vector (difference between desired and predicted signals).
#             Shape: (L,)
#         wts : numpy.ndarray
#             Weight history matrix tracking the evolution of weights over time.
#             Shape: (N, L) where each column i contains the weight vector after
#             processing sample i.
    
#     Notes
#     -----
#     - Learning rate (mu) is fixed at 1e-5
#     - Normalization factor is set to 1 (no normalization applied)
#     - The input row is flipped before computing the prediction to match the
#       expected filter structure
#     - Weight updates follow the LMS gradient descent rule:
#       w(n+1) = w(n) + (mu / norm_factor) * e(n) * x(n)
    
#     Examples
#     --------
#     >>> x = np.random.randn(1000, 10)  # 1000 samples, filter order 10
#     >>> d = np.random.randn(1000)       # Desired signal
#     >>> y, e, wts = my_pyt_lms(x, d, nSteps=0, N=10)
#     >>> print(f"Final MSE: {np.mean(e**2)}")
#     """
#     # Get the length of the input signal
#     L = len(x)
    
#     # Initialize weights to zeros if not provided
#     if w is None:
#         w = np.zeros(N)
    
#     # Initialize output arrays
#     y = np.zeros(L)  # Prediction vector
#     e = np.zeros(L)  # Error vector
#     # wts = np.zeros((N, L))  # Weight history matrix (N × L)
#     wts = np.zeros((L, N))  # Weight history matrix (L × N)
    
#     # LMS algorithm parameters
#     mu = 1e-5  # Learning rate
#     norm_factor = 1  # Normalization factor
    
#     # Main LMS iteration loop
#     for i in range(L):
#         # Flip the input row to create the input vector
#         x_n = x[i, ::-1]
        
#         # Compute prediction using current weights
#         y[i] = w.T @ x_n
        
#         # Calculate error
#         e[i] = d[i] - y[i]
        
#         # Update weights using LMS gradient descent (if not locked)
#         if not lock_coeff:
#             w = w + (mu / norm_factor) * e[i] * x_n
        
#         # Store current weight vector in history
#         # wts[:, i] = w
#         wts[i, :] = w
    
#     return y, e, wts


def my_pyt_lms(x_input, d_target, N, w=None, lock_coeff=False):

#   X = x_input.values
#   d = d_target.values

  X = x_input
  d = d_target

  L = len(X)  # Number of data points

  # Initialize outputs
  if w is None:
      w = np.zeros(N)  # Default: Initialize weights to zero

  e = np.zeros(L)
  y = np.zeros(L)
  wts = np.zeros((L, N))
  mu = 1e-5  # Learning rate

  # LMS Algorithm Iteration
  for n in range(L):
      # Input segment (row of lagged features)
      x_n = X[n, :]

      # Calculate predicted output
      y[n] = np.dot(w, x_n)

    #   # Calculate error
    #   e[n] = d[n] - y[n]

      # Update weights if lock_coeff is False
      if not lock_coeff:
          # Calculate error
          e[n] = d[n] - y[n]
          w = w + (mu) * e[n] * x_n  # Simplified update rule

      # Store weights for analysis
      wts[n, :] = w

  return y, e, wts



# i had to use the LMS code from the colab bc here it was giving me shape errors.