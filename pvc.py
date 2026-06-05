import numpy as np

def resolver_pvc1_1d(N):

    dx = 1.0 / N
    n_incognitas = N - 1
    
    A = np.zeros((n_incognitas, n_incognitas))
    b = np.zeros(n_incognitas)
    
    diag_principal = -(2.0 / (dx**2) + 1.0)
    diag_secundaria = 1.0 / (dx**2)
    
    for i in range(n_incognitas):
        A[i, i] = diag_principal
        if i > 0:
            A[i, i-1] = diag_secundaria
        if i < n_incognitas - 1:
            A[i, i+1] = diag_secundaria
            
    b[-1] = -(1.0 / (dx**2)) * 1.0 
    
    y_internos = np.linalg.solve(A, b)
    
    y_completo = np.zeros(N + 1)
    y_completo[0] = 0.0
    y_completo[-1] = 1.0
    y_completo[1:-1] = y_internos
    
    x = np.linspace(0, 1, N + 1)
    return x, y_completo

def resolver_pvc2_2d(N, f_val=4.0):
    h = 1.0 / N
    n_internos = N - 1
    num_equacoes = n_internos * n_internos
    
    A = np.zeros((num_equacoes, num_equacoes))
    b = np.zeros(num_equacoes)
    
    def index_k(i, j):
        return j * n_internos + i
        
    for j in range(n_internos):
        for i in range(n_internos):
            k = index_k(i, j)
            
            A[k, k] = -2.0 * (1.0/(h**2) + 1.0/(h**2))
            
            if i > 0:
                A[k, index_k(i-1, j)] = 1.0 / (h**2)
                
            if i < n_internos - 1:
                A[k, index_k(i+1, j)] = 1.0 / (h**2)
                
            if j > 0:
                A[k, index_k(i, j-1)] = 1.0 / (h**2)
                
            if j < n_internos - 1:
                A[k, index_k(i, j+1)] = 1.0 / (h**2)
                
            b[k] = f_val
            
    u_internos = np.linalg.solve(A, b)
    
    U_completa = np.zeros((N + 1, N + 1))
    for j in range(n_internos):
        for i in range(n_internos):
            U_completa[j+1, i+1] = u_internos[index_k(i, j)]
            
    return U_completa