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