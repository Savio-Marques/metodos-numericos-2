import numpy as np

def potencia_regular(A, v0, tolerancia=1e-6, max_iter=1000):
    lambda_novo = 0.0
    vk_novo = np.copy(v0)
    iteracao = 0
    
    while iteracao < max_iter:
        iteracao += 1
        lambda_velho = lambda_novo
        vk_velho = np.copy(vk_novo)
        
        norma = np.sqrt(np.dot(vk_velho.T, vk_velho))
        x1_velho = vk_velho / norma
        
        vk_novo = np.dot(A, x1_velho)
        lambda_novo = np.dot(x1_velho.T, vk_novo)
        
        if lambda_novo != 0:
            erro = abs((lambda_novo - lambda_velho) / lambda_novo)
            if erro < tolerancia:
                break
                
    return lambda_novo, x1_velho, iteracao

def potencia_inverso(A, v0, tolerancia=1e-6, max_iter=1000):
    lambda_novo = 0.0
    vk_novo = np.copy(v0)
    iteracao = 0
    
    while iteracao < max_iter:
        iteracao += 1
        lambda_velho = lambda_novo
        vk_velho = np.copy(vk_novo)
        
        norma = np.sqrt(np.dot(vk_velho.T, vk_velho))
        x1_velho = vk_velho / norma
        
        vk_novo = np.linalg.solve(A, x1_velho)
        
        lambda_novo = np.dot(x1_velho.T, vk_novo)
        
        if lambda_novo != 0:
            erro = abs((lambda_novo - lambda_velho) / lambda_novo)
            if erro < tolerancia:
                break
                
    lambda_n = 1.0 / lambda_novo
    
    return lambda_n, x1_velho, iteracao

def potencia_deslocamento(A, v0, mu, tolerancia=1e-6, max_iter=1000):
    n = A.shape[0]
    
    I = np.eye(n)
    A_hat = A - mu * I
    
    lambda_hat, x_hat, iteracao = potencia_inverso(A_hat, v0, tolerancia, max_iter)
    
    lambda_i = lambda_hat + mu
    
    return lambda_i, x_hat, iteracao

def matriz_householder(A, i):
    n = A.shape[0]
    w = np.zeros(n)
    
    w[i+1:] = A[i+1:, i]
    Lw = np.linalg.norm(w)
    
    if Lw == 0:
        return np.eye(n)
        
    w_linha = np.zeros(n)
    
    sinal = 1 if w[i+1] >= 0 else -1
    w_linha[i+1] = -sinal * Lw 
    
    N = w - w_linha
    norma_N = np.linalg.norm(N)
    
    if norma_N == 0:
        return np.eye(n)
        
    n_vec = N / norma_N
    n_vec = n_vec.reshape(-1, 1)
    
    H = np.eye(n) - 2 * (np.dot(n_vec, n_vec.T))
    return H

def householder_tridiagonal(A):
    n = A.shape[0]
    A_k = np.copy(A)
    
    for i in range(n - 2):
        H = matriz_householder(A_k, i)
        A_k = np.dot(np.dot(H, A_k), H) 
        
    return A_k


def matriz_rotacao_givens(a, b):
    if b == 0:
        return 1.0, 0.0
        
    if abs(b) > abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1.0 + tau**2)
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / np.sqrt(1.0 + tau**2)
        s = c * tau
    return c, s

def decomposicao_qr_givens(A):
    n = A.shape[0]
    R = np.copy(A)
    Q_t = np.eye(n)
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            if R[j, i] != 0:
                c, s = matriz_rotacao_givens(R[i, i], R[j, i])
                
                linha_i = c * R[i, :] - s * R[j, :]
                linha_j = s * R[i, :] + c * R[j, :]
                R[i, :] = linha_i
                R[j, :] = linha_j
                
                q_i = c * Q_t[i, :] - s * Q_t[j, :]
                q_j = s * Q_t[i, :] + c * Q_t[j, :]
                Q_t[i, :] = q_i
                Q_t[j, :] = q_j
                
    return Q_t.T, R

def metodo_qr(A, tol=1e-6, max_iter=1000):
    A_k = np.copy(A)
    iteracao = 0
    
    while iteracao < max_iter:
        iteracao += 1
        Q, R = decomposicao_qr_givens(A_k)
        
        A_k = np.dot(R, Q)
        
        subdiag = np.diag(A_k, k=-1)
        if np.all(np.abs(subdiag) < tol):
            break
            
    autovalores = np.diag(A_k)
    return autovalores, iteracao