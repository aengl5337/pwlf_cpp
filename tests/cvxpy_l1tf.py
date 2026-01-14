import numpy as np
import cvxpy as cp
import scipy as scipy
import cvxopt as cvxopt
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter #, find_peaks
import pwlf
from rdp import rdp
import time

def moving_average(vec, window=3):
    """Return centered moving average of 1D array `vec` with odd `window`.
    Edges are handled with edge-padding so output length equals input length.
    """
    vec = np.asarray(vec, dtype=float)
    if window <= 1:
        return vec.copy()
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window) # Results in uniform weights
    # Pad the input vector at both ends with edge values, so that convolution 'valid' mode returns original length
    vec_padded = np.pad(vec, pad_width=pad, mode='edge')
    # 'valid' on the padded array returns the same length as original
    return np.convolve(vec_padded, kernel, mode='valid')

def preprocess(Y):
    """ Preprocess data by removing duplicate deflection points and smoothing loading data. """
    t = Y[:, 0]
    y = Y[:, 1]

    # Check for duplicate deflection points (i.e. where diff=0)
    h = np.diff(t)
    duplicates = np.equal(h, 0)
    if np.any(duplicates):
        # Remove duplicate deflection points
        unique_indices = np.where(~duplicates)[0]
        t = t[unique_indices]
        y = y[unique_indices]
    
    Y_rough = np.column_stack((t, y))
    n = Y_rough.shape[0]
    # # Smooth loading data with moving average filter
    # y = moving_average(y, window=3)

    
    # Use Savitzky-Golay filter to smooth while preserving edges
    # window_length must be odd; adjust based on your data density
    window_length = n // 80
    if window_length % 2 == 0:
        window_length += 1  # make it odd
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=7)

    Y_smooth = np.column_stack((t, y_smooth))

    return Y_smooth, Y_rough

# Form second difference matrix. (for unevenly-spaced data)
def secondDiff_arbSpacing(t, assumeEvenSpacing=False):
    n = t.size
    e = np.ones((1, n-2))

    if assumeEvenSpacing:
        diagonals = [e, -2*e, e]
    else:
        h = np.diff(t) # step sizes
        scaling = 2*np.average(h)**2 # scale to make comparable to even spacing case but still preserve relative scale between rows
        e = scaling*e
        diagonals = [ e   / (h[1:]  * (h[1:] + h[:-1])),
                     -e / (h[1:]  *  h[:-1]),
                      e / (h[:-1] * (h[1:] + h[:-1]))]

        # diagonals = [ e   /  h[1:],
        #              -e   *  (1/h[1:] + 1/h[:-1]),
        #               e   /   h[:-1]]

    return scipy.sparse.diags_array(diagonals, offsets=range(3), shape=(n-2, n))

def lambdaMax_l1tf(D, y):
    """ Compute lambda_max as defined in L1 trend filtering paper. """
    Dy = D * y # Note, * more efficient than D @ y for sparse D
    DDt_inv = np.linalg.inv((D @ D.T).toarray())
    lambda_max = np.max(np.abs(DDt_inv @ Dy))

    return lambda_max

def RDP(points, target_count=4):
    """
    Finds exactly `target_count` points (Start, Kink1, Kink2, ... End)
    by adaptively sweeping epsilon from coarse to fine.
    """
    
    # 1. Determine the scale of the data
    # We use the Y-axis span (max - min) to set our search bounds.
    # This removes the need for magic numbers like "2.0".
    y_span = np.ptp(points[:, 1])  # Peak-to-peak (max - min)
    
    # 2. Define search parameters
    # Start searching at 20% of the total height (very coarse)
    start_eps = y_span * 0.2
    # Stop if we get too fine (e.g., 0.1% of height - likely noise)
    min_eps = y_span * 0.001
    # Number of steps in our sweep
    steps = 100
    
    best_trajectory = None
    found_epsilon = None

    # 3. The Sweep (Logarithmic is usually better for scale)
    # We sweep downwards: from "Loose fit" -> "Tight fit"
    search_space = np.logspace(np.log10(start_eps), np.log10(min_eps), steps)

    for eps in search_space:
        simplified = rdp(points, epsilon=eps)
        
        # We want exactly the target count (Start + 2 Kinks + End = 4)
        # If we find MORE than 4, we've gone too deep and hit noise.
        # We return the last valid result that had <= 4 points.
        if len(simplified) == target_count:
            return np.array(simplified), eps
        
        elif len(simplified) > target_count:
            # We missed the exact target and jumped to too many points.
            # This happens if the kinks are small or noise is high.
            # In this case, the previous iteration was likely the best approximation.
            print(f"Warning: Jumped from {len(best_trajectory)} to {len(simplified)} points.")
            break
            
        # Store the current best attempt
        best_trajectory = simplified
        found_epsilon = eps

    # If we exit the loop without hitting the target, return what we have
    return np.array(best_trajectory) if best_trajectory is not None else points, found_epsilon

def plot_trend(data_dict, plambda):
    # Plot properties.
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # font = {'weight' : 'normal',
    #         'size'   : 16}
    # plt.rc('font', **font)

    # Plot trend estimates with original signal.
    plt.figure(figsize=(10, 6))
    for key, Y in data_dict.items():
        plt.plot(Y[:, 0], Y[:, 1], linewidth=2.0, label=key)

    plt.legend()
    plt.xlabel('Deflection')
    plt.ylabel('Loading')
    plt.title(f'Trend Filtering Results (λ={plambda:.3f} × λ_max)')
    plt.grid(True, alpha=0.3)
    plt.show()

def CVXPY(Y, D=None, plambda=None, objective='l1tf', weighting=None, verbose=False):
    """ https://www.cvxpy.org/examples/applications/l1_trend_filter.html """
    
    if objective not in ['l1tf', 'hp', 'l1l1']:
        raise ValueError("Invalid objective function specified.")
    
    # Extract data vectors.
    if Y.ndim == 1:
        n = Y.size
        t = np.arange(1,n+1)
        y = Y
    else:
        n = Y.shape[0]
        t = Y[:, 0]
        y = Y[:, 1]

    # Form second difference matrix.
    if D is None:
        D = secondDiff_arbSpacing(t, assumeEvenSpacing=True)
    
    # Set regularization parameter (lambda).
    if plambda is None:
        plambda = 0.01
    
    lambda_max = lambdaMax_l1tf(D,y)
    print("Lambda max: {}".format(lambda_max))

    vlambda = plambda*lambda_max
    print("Using lambda: {}".format(vlambda))

    # Solve trend filtering problem.
    yhat = cp.Variable(shape=n)

    # Prepare weighting vector. Options: None, 'gaussian', or an array-like of length n.
    if weighting is None:
        w = np.ones(n)
    elif isinstance(weighting, str) and weighting == 'gaussian':
        # center near t=0 if 0 inside the range, otherwise use mean(t)
        center = 0.0 if (t.min() <= 0.0 <= t.max()) else np.mean(t)
        span = t.max() - t.min()
        sigma = 0.15 * span if span > 0 else 1.0
        amplitude = 5.0  # peak additional weighting at center
        w = 1.0 + amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)
    elif isinstance(weighting, (list, np.ndarray)):
        w = np.asarray(weighting, dtype=float)
        if w.shape[0] != n:
            raise ValueError('weighting vector must have length n')
    else:
        raise ValueError("Invalid weighting option specified.")
        
    r = cp.multiply(w, y - yhat)
    if objective == 'l1tf':  # L1 Trend Filter        
        obj = cp.Minimize(0.5 * cp.sum_squares(r)
                          + vlambda * cp.norm(D @ yhat, 1))
    elif objective == 'hp':  # Hodrick-Prescott filter
        obj = cp.Minimize(0.5 * cp.sum_squares(r)
                          + vlambda * cp.sum_squares(D @ yhat))
    elif objective == 'l1l1':  # L1-L1 Trend Filter
        obj = cp.Minimize(0.5 * cp.norm(r, 1)
                          + vlambda * cp.norm(D @ yhat, 1))
        
    prob = cp.Problem(obj)

    # ECOS and SCS solvers fail to converge before
    # the iteration limit. Use CVXOPT instead.
    prob.solve(solver=cp.CVXOPT, verbose=verbose)
    # prob.solve(solver = cp.SCS, verbose=True)
    print('Solver status: {}'.format(prob.status))

    # Check for error.
    if prob.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")

    print("optimal objective value: {}".format(obj.value))

    Yhat = np.column_stack((t, yhat.value))

    return Yhat, vlambda

def PWLF(Y, initial_breaks=None, weights=None):
    """ Piecewise Linear Fit with 3 segments (2 kinks) using pwlf package. """
    if Y.ndim == 1:
        n = Y.size
        t = np.arange(1,n+1)
        y = Y
    else:
        n = Y.shape[0]
        t = Y[:, 0]
        y = Y[:, 1]

    # Prepare weighting vector. Options: None, 'gaussian', 'gaussian_breaks', or an array-like of length n.
    if weights is None:
        w = np.ones(n)
    
    elif isinstance(weights, str):
        if  weights == 'gaussian':
            # center near t=0 if 0 inside the range, otherwise use mean(t)
            center = 0.0 if (t.min() <= 0.0 <= t.max()) else np.mean(t)
            span = t.max() - t.min()
            sigma = 0.15 * span if span > 0 else 1.0
            amplitude = 5.0  # peak additional weighting at center
            w = 1.0 + amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)

        elif weights == 'gaussian_breaks':
            if initial_breaks is None:
                raise ValueError("initial_breaks must be provided for 'gaussian_breaks' weighting.")
            w = np.ones(n)
            amplitude = 10.0  # peak additional weighting at breakpoints
            sigma = abs(initial_breaks[1] - initial_breaks[0])/2  # narrow Gaussian around breakpoints, and ensure don't overlap too much
            for brk in initial_breaks:
                w += amplitude * np.exp(-0.5 * ((t - brk) / sigma) ** 2)
        
        else:
            raise ValueError("Invalid weights option specified.")
        
    elif isinstance(weights, (list, np.ndarray)): # Manual weight vector
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != n:
            raise ValueError('weights vector must have length n')
    
    else:
        raise ValueError("Invalid weights option specified.")
    
    # initialize piecewise linear fit with your t and y data
    my_pwlf = pwlf.PiecewiseLinFit(t, y, weights=w)

    # fit the data for 2 breakpoints if have a guess, which leads to 3 line segments intrinsically (start/end included automatically). else fit explicitly for 3 line segments
    if initial_breaks is not None:
        # res = my_pwlf.fit_with_breaks(initial_breaks) # fit with user-defined breakpoints, these are not varied
        res = my_pwlf.fit_guess(initial_breaks)  # fit with user-defined number of line segments, breakpoints are optimized
    else:
        res = my_pwlf.fit(3)

    # predict for the determined points
    that = np.linspace(min(t), max(t), num=10000)
    yhat = my_pwlf.predict(that)

    return np.column_stack((that, yhat)), my_pwlf

def main():
    # Load Loading vs Displacement characteristic data for RHC
    Y = np.loadtxt('rhc_deflection.txt')
    Y, Y_rough = preprocess(Y) # Y is now smoothed
    t = Y[:, 0]
    y = Y[:, 1]

    data_dict = {'Original Signal': Y_rough, 'Smoothed Signal': Y}

    # Set standard lambda proportion
    plambda = 0.01

    # Define labels for different objectives
    labels = {
            'l1tf': 'L1 Trend Filter',
            'l1l1': 'L1-L1 Trend Filter',
            'hp': 'Hodrick-Prescott Filter',
            }

    # # CVXPY, EVENLY SPACED ASSUMPTION (for second difference matrix)
    # print("CVXPY, Assuming even spacing for second difference matrix...")
    # Deven = secondDiff_arbSpacing(t, assumeEvenSpacing=True)
    # Deven_y = Deven @ y
    # Deven_y = np.pad(Deven_y, (1,1), 'edge')  # Pad to match original length (since 2nd difference is a 3 element discrete kernel) for plotting
    # # data_dict['2nd Difference, even spacing'] = np.column_stack((t, Deven_y))
    
    # for objective in ['l1tf']: #, 'l1l1', 'hp'
    #     print(f"Fitting with CVXPY {labels[objective]}...")
    #     Yhat, vlambda = CVXPY(Y, D=Deven, plambda=plambda, objective=objective)
    #     data_dict[f'CVXPY {labels[objective]}'] = Yhat

    # CVXPY, UNEVENLY SPACED CASE
    # print("CVXPY, Accounting for uneven spacing...")
    # Darb = secondDiff_arbSpacing(t, assumeEvenSpacing=False)
    # Darb_y = Darb @ y
    # Darb_y = np.pad(Darb_y, (1,1), 'edge')  # Pad to match original length for plotting
    # data_dict['2nd Difference, arb spacing'] = np.column_stack((t, Darb_y))

    # for objective in ['l1tf']:
    #     label = labels[objective]+' (arb spacing)'
    #     print(f"Fitting with CVXPY {label}...")
    #     Yhat, vlambda = CVXPY(Y, D=Darb, plambda=plambda, objective=objective, weighting=None)
    #     data_dict[f'CVXPY {label}'] = Yhat

    # # CVXPY, UNEVENLY SPACED CASE, L1 Trend Filter pre-processed with H-P filter (Whittaker-Eilers smoothing)
    # print("CVXPY L1 Trend Filter on H-P pre-smoothed data...")
    # Yhp, vlambda = CVXPY(Y, D=Darb, plambda=plambda, objective='hp')
    # Yhp = data_dict['CVXPY Hodrick-Prescott Filter (arb spacing)']
    # Yhat, vlambda = CVXPY(Yhp, D=Darb, plambda=plambda, objective='l1tf')
    # data_dict['CVXPY L1 Trend Filter on H-P smoothed data'] = Yhat

    # # CVXPY, UNEVENLY SPACED CASE, L1 trend filter with and without gaussian weighting
    # for weighting_option in ['gaussian']: # None, 
    #     label = 'with Gaussian Weighting' if weighting_option == 'gaussian' else 'without Weighting'
    #     print(f"Fitting with CVXPY L1 Trend Filter {label}...")
    #     Yhat, vlambda = CVXPY(Y, D=Darb, plambda=plambda, objective='l1tf', weighting=weighting_option)
    #     data_dict[f'CVXPY L1 Trend Filter {label}'] = Yhat

    # COMPARE WITH RDP SIMPLIFICATION
    print("Fitting with RDP simplification to find kinks...")
    target_points = 4  # Start, Kink1, Kink2, End
    # Search for the optimal epsilon
    simplified_trajectory, optimal_eps = RDP(Y, target_count=target_points)
    print(f"Optimization Complete.")
    print(f"Optimal Epsilon: {optimal_eps:.4f}")
    print(f"Points Found: {len(simplified_trajectory)}")
    # Access the kinks (excluding start/end)
    if len(simplified_trajectory) >= 3:
        kinks = simplified_trajectory[1:-1]
        print("Kinks found at:\n", kinks)
        data_dict['RDP Detected Kinks'] = simplified_trajectory
    else:
        print("Could not isolate distinct kinks. Data may be too linear or too noisy.")

    # # COMPARE WITH PWLF (Piecewise linear fit with 2 kinks)
    # print("Fitting with PWLF (2 kinks)...")
    # start_time = time.perf_counter()
    # Yhat_pwlf, pwlf_model = PWLF(Y)
    # end_time = time.perf_counter()
    # execution_time_1 = end_time - start_time
    # print(f"PWLF 3 segment fit: {execution_time_1:.6f} seconds")
    # data_dict['PWLF 3 segment fit'] = Yhat_pwlf

    # PWLF WITH RDP-INITIALIZED BREAKPOINTS
    if len(simplified_trajectory) == target_points:
        print("Fitting PWLF using RDP-detected kinks as initial breakpoints...")
        initial_breaks = simplified_trajectory[1:-1, 0]  # t-values of detected kinks (exclude start/end)... should be 2 floats
        start_time = time.perf_counter()
        Yhat_pwlf_rdp, pwlf_model_rdp = PWLF(Y, initial_breaks=initial_breaks, weights=None)
        end_time = time.perf_counter()
        execution_time_1 = end_time - start_time
        print(f"PWLF 2 kink fit with RDP init: {execution_time_1:.6f} seconds")
        data_dict['PWLF 2 kink fit with RDP init'] = Yhat_pwlf_rdp
    else:
        print("Skipping PWLF with RDP breakpoints due to insufficient points found.")

    # PWLF WITH RDP-INITIALIZED BREAKPOINTS and weights
    if len(simplified_trajectory) == target_points:
        print("Fitting PWLF using RDP-detected kinks as initial breakpoints...")
        initial_breaks = simplified_trajectory[1:-1, 0]  # t-values of detected kinks (exclude start/end)... should be 2 floats
        start_time = time.perf_counter()
        Yhat_pwlf_rdp, pwlf_model_rdp = PWLF(Y, initial_breaks=initial_breaks, weights='gaussian_breaks')
        end_time = time.perf_counter()
        execution_time_1 = end_time - start_time
        print(f"PWLF 2 kink fit with RDP init (and weighting): {execution_time_1:.6f} seconds")
        data_dict['PWLF 2 kink fit with RDP init (and weighting)'] = Yhat_pwlf_rdp
    else:
        print("Skipping PWLF with RDP breakpoints due to insufficient points found.")


    

    # PLOT RESULTS
    plot_trend(data_dict, plambda=plambda)

if __name__ == '__main__':
    main()