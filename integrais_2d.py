import numpy as np

def gauss_legendre_2d_3pontos(f_uv):
    pontos = [-np.sqrt(3/5), 0.0, np.sqrt(3/5)]
    pesos = [5/9, 8/9, 5/9]
    
    integral = 0.0
    
    for i in range(3):
        for j in range(3):
            u = pontos[i]
            v = pontos[j]
            peso_duplo = pesos[i] * pesos[j]
            
            integral += peso_duplo * f_uv(u, v)
            
    return integral