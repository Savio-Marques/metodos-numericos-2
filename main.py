import numpy as np
import autovalores as autov

def relatorio_matriz(nome, A, mu_chute):
    n = A.shape[0]
    v0 = np.ones(n)
    
    print(f"\n{'-'*40}")
    print(f" MATRIZ {nome} ({n}x{n})")
    print(f"{'-'*40}")
    
    l_max, x_max, it1 = autov.potencia_regular(A, v0)
    print(f"[REGULAR] Maior Autovalor: {l_max:.6f} (Iterações: {it1})")
    
    l_min, x_min, it2 = autov.potencia_inverso(A, v0)
    print(f"[INVERSO] Menor Autovalor: {l_min:.6f} (Iterações: {it2})")
    
    l_mid, x_mid, it3 = autov.potencia_deslocamento(A, v0, mu=mu_chute)
    print(f"[SHIFT μ={mu_chute}] Autovalor Pescado: {l_mid:.6f} (Iterações: {it3})")
    
if __name__ == "__main__":
    A1 = np.array([
        [5.0, 2.0, 1.0],
        [2.0, 3.0, 1.0],
        [1.0, 1.0, 2.0]
    ])
    
    A2 = np.array([
        [-14.0,   1.0,  -2.0],
        [  1.0,  -1.0,   1.0],
        [ -2.0,   1.0, -11.0]
    ])
    
    A3 = np.array([
        [40.0,  8.0,  4.0,  2.0, 1.0],
        [ 8.0, 30.0, 12.0,  6.0, 2.0],
        [ 4.0, 12.0, 20.0,  1.0, 2.0],
        [ 2.0,  6.0,  1.0, 25.0, 4.0],
        [ 1.0,  2.0,  2.0,  4.0, 5.0]
    ])

    relatorio_matriz("A1", A1, mu_chute=3.0)
    
    relatorio_matriz("A2", A2, mu_chute=-10.0)
    
    relatorio_matriz("A3", A3, mu_chute=20.0)