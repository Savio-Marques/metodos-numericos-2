import numpy as np

def resolver_mef_1d(N):

    h = 1.0 / N
    n_nos = N + 1
    K = np.zeros((n_nos, n_nos))
    
    ke = (1.0/h) * np.array([[1, -1], [-1, 1]]) + (h/6.0) * np.array([[2, 1], [1, 2]])
    
    for e in range(N):
        i, j = e, e + 1
        K[i:j+1, i:j+1] += ke
        
        K_reduzida = K[1:N, 1:N]
    b = np.zeros(N - 1)
    b[-1] -= K[N-1, N] * 1.0
    
    u_internos = np.linalg.solve(K_reduzida, b)
    
    y = np.zeros(n_nos)
    y[0] = 0.0
    y[N] = 1.0
    y[1:N] = u_internos
    
    x = np.linspace(0, 1, n_nos)
    return x, y

def resolver_mef_geral(B, C, D, a, b, u_a, u_b, N):
    h = (b - a) / N
    n_nos = N + 1
    K = np.zeros((n_nos, n_nos))
    F = np.zeros(n_nos)
    
    ke_diff = (1.0/h) * np.array([[1, -1], [-1, 1]])
    ke_conv = -B * np.array([[-0.5, 0.5], [-0.5, 0.5]])
    ke_reac = -C * (h/6.0) * np.array([[2, 1], [1, 2]])
    
    ke = ke_diff + ke_conv + ke_reac
    fe = -D * (h/2.0) * np.array([1, 1])
    
    for e in range(N):
        i, j = e, e + 1
        K[i:j+1, i:j+1] += ke
        F[i:j+1] += fe
        
    K[0, :] = 0
    K[0, 0] = 1.0
    F[0] = u_a
    
    K[-1, :] = 0
    K[-1, -1] = 1.0
    F[-1] = u_b
    
    u = np.linalg.solve(K, F)
    x = np.linspace(a, b, n_nos)
    
    return x, u