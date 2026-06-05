import numpy as np

def runge_kutta_3(F, S_i, t_i, dt):
    F1 = F(S_i, t_i)
    
    S_meio = S_i + (dt / 2.0) * F1
    
    F2 = F(S_meio, t_i + dt / 2.0)
    
    S_bar_prox = S_i + dt * (-F1 + 2.0 * F2)
    
    F3 = F(S_bar_prox, t_i + dt)
    
    S_prox = S_i + dt * ((1.0 / 6.0) * F1 + (4.0 / 6.0) * F2 + (1.0 / 6.0) * F3)
    
    return S_prox

def resolver_pvi(F, S0, t0, dt, condicao_parada):
    t = t0
    S = np.array(S0, dtype=float)
    
    historico_t = [t]
    historico_S = [S]
    
    while not condicao_parada(t, S):
        S = runge_kutta_3(F, S, t, dt)
        t += dt
        
        historico_t.append(t)
        historico_S.append(S)
        
    return np.array(historico_t), np.array(historico_S)