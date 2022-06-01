import numpy as np

def payoff_barrier(x,K,b,putorcall):
    num_paths, time_steps = np.shape(x)
    payoff = np.zeros(num_paths)
    for i in range(num_paths):
        x_i = x[i,:]
    if np.max(x_i) < b:
        if putorcall == "call":
            payoff[i] = -np.maximum(x_i[-1] - K, 0.0)
        elif putorcall == "put":
            payoff[i] = -np.maximum(K - x_i[-1], 0.0)
    else:
        payoff[i] = 0
    return payoff
