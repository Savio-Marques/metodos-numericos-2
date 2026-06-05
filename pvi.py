import numpy as np

def runge_kutta_4(F, S_i, t_i, dt):
    F1 = F(S_i, t_i)
    S2 = S_i + (dt / 2.0) * F1
    
    F2 = F(S2, t_i + dt / 2.0)
    S3 = S_i + (dt / 2.0) * F2
    
    F3 = F(S3, t_i + dt / 2.0)
    S4 = S_i + dt * F3
    
    F4 = F(S4, t_i + dt)
    
    S_prox = S_i + (dt / 6.0) * (F1 + 2.0 * F2 + 2.0 * F3 + F4)
    return S_prox

def preditor_corretor_4(F, S0, t0, dt, condicao_parada, tol=1e-7, max_iter=100):
    t = t0
    S = np.array(S0, dtype=float)
    
    historico_t = [t]
    historico_S = [S]
    historico_F = [F(S, t)]
    
    for _ in range(3):
        if condicao_parada(t, S):
            break
        S = runge_kutta_4(F, S, t, dt)
        t += dt
        
        historico_t.append(t)
        historico_S.append(S)
        historico_F.append(F(S, t))
        
    while not condicao_parada(historico_t[-1], historico_S[-1]):
        t_i = historico_t[-1]
        t_prox = t_i + dt
        
        Fi   = historico_F[-1]
        Fi_1 = historico_F[-2]
        Fi_2 = historico_F[-3]
        Fi_3 = historico_F[-4]
        Si   = historico_S[-1]
        
        S_pred = Si + (dt / 24.0) * (55.0 * Fi - 59.0 * Fi_1 + 37.0 * Fi_2 - 9.0 * Fi_3)
        
        S_corr_ant = S_pred
        for k in range(max_iter):
            F_prox = F(S_corr_ant, t_prox)
            
            S_corr_novo = Si + (dt / 24.0) * (9.0 * F_prox + 19.0 * Fi - 5.0 * Fi_1 + Fi_2)
            
            erro_relativo = np.max(np.abs(S_corr_novo - S_corr_ant) / (np.abs(S_corr_novo) + 1e-15))
            if erro_relativo < tol:
                break
                
            S_corr_ant = S_corr_novo
            
        S_novo = S_corr_novo
        
        historico_t.append(t_prox)
        historico_S.append(S_novo)
        historico_F.append(F(S_novo, t_prox))
        
    return np.array(historico_t), np.array(historico_S)