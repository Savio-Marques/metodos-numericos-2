import numpy as np
import autovalores as autov

if __name__ == "__main__":
    print("="*50)
    print(" CÁLCULO DE AUTOVALORES (MÉTODO DA POTÊNCIA)")
    print("="*50)

    A1 = np.array([
        [5.0, 2.0, 1.0],
        [2.0, 3.0, 1.0],
        [1.0, 1.0, 2.0]
    ])
    
    A2 = np.array([
        [40.0,  8.0,  4.0,  2.0, 1.0],
        [ 8.0, 30.0, 12.0,  6.0, 2.0],
        [ 4.0, 12.0, 20.0,  1.0, 2.0],
        [ 2.0,  6.0,  1.0, 25.0, 4.0],
        [ 1.0,  2.0,  2.0,  4.0, 5.0]
    ])

    v0_A1 = np.ones(3)
    v0_A2 = np.ones(5)

    lambda1, x1, iters1 = autov.potencia_regular(A1, v0_A1)
    print("\n[Matriz A1 (3x3)]")
    print(f"Iterações necessárias: {iters1}")
    print(f"Autovalor Dominante  : {lambda1:.5f}")
    print(f"Autovetor Associado  : {x1}")

    lambda2, x2, iters2 = autov.potencia_regular(A2, v0_A2)
    print("\n[Matriz A2 (5x5)]")
    print(f"Iterações necessárias: {iters2}")
    print(f"Autovalor Dominante  : {lambda2:.5f}")
    print(f"Autovetor Associado  : {x2}")
    print("="*50)