import numpy as np

def gauss_legendre_3d_3pontos(f_uvw):
    pontos = [-np.sqrt(3/5), 0.0, np.sqrt(3/5)]
    pesos = [5/9, 8/9, 5/9]
    
    integral = 0.0
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = pontos[i]
                v = pontos[j]
                w = pontos[k]
                
                peso_triplo = pesos[i] * pesos[j] * pesos[k]
                
                integral += peso_triplo * f_uvw(u, v, w)
                
    return integral