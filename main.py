import numpy as np
import pvi

def F_PVI_2(S, t):
    v = S[0]
    y = S[1]
    
    g = 10.0
    k = 0.25
    m = 2.0
    
    dv_dt = -g - (k / m) * v
    dy_dt = v
    return np.array([dv_dt, dy_dt])

def paragem_no_mar(t, S):
    y = S[1]
    return y <= 0

if __name__ == "__main__":
    print("="*65)
    print(" TAREFA 18: SOLUÇÃO DE PVI COM PREDITOR-CORRETOR 4ª ORDEM")
    print("="*65)
    
    t0 = 0.0
    v0 = 5.0     
    y0 = 200.0   
    S0 = [v0, y0]
    
    valores_dt = [0.1, 0.01, 0.001, 0.0001]
    
    for dt in valores_dt:
        t_hist, S_hist = pvi.preditor_corretor_4(F_PVI_2, S0, t0, dt, paragem_no_mar)
        
        v_hist = S_hist[:, 0]
        y_hist = S_hist[:, 1]
        
        idx_max = np.argmax(y_hist)
        y_max = y_hist[idx_max]
        t_max = t_hist[idx_max]
        t_total = t_hist[-1]
        v_impacto = v_hist[-1]
        
        print(f"\n[ Resultados para dt = {dt} ]")
        print(f"a) Altura máxima alcançada (y_max) : {y_max:.5f} m")
        print(f"b) Tempo até a altura máxima (t_max): {t_max:.5f} s")
        print(f"c) Tempo total até o impacto (t_tot): {t_total:.5f} s")
        print(f"d) Velocidade de impacto no mar     : {v_impacto:.5f} m/s")

    print("\n" + "="*65)