
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