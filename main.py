import numpy as np
import autovalores as autov

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    
    print("="*60)
    print(" TAREFA #13 E #15: HOUSEHOLDER E MÉTODO QR")
    print("="*60)
    
    A = np.array([
        [40.0,  8.0,  4.0,  2.0, 1.0],
        [ 8.0, 30.0, 12.0,  6.0, 2.0],
        [ 4.0, 12.0, 20.0,  1.0, 2.0],
        [ 2.0,  6.0,  1.0, 25.0, 4.0],
        [ 1.0,  2.0,  2.0,  4.0, 5.0]
    ])
    
    print("\n1. MATRIZ ORIGINAL (A):")
    print(A)
    
    A_tridiag = autov.householder_tridiagonal(A)
    
    print("\n2. MATRIZ TRIDIAGONALIZADA (Householder) - TAREFA #13:")
    print("(Note como os zeros aparecem nos cantos!)")
    print(A_tridiag)
    
    autovalores_finais, iters = autov.metodo_qr(A_tridiag)
    
    autovalores_finais = np.sort(autovalores_finais)[::-1]
    
    print(f"\n3. RESULTADO DO MÉTODO QR - TAREFA #15:")
    print(f"O Método convergiu em {iters} iterações.")
    print("Todos os Autovalores Encontrados:")
    for i, val in enumerate(autovalores_finais, 1):
        print(f" λ{i} = {val:.6f}")
        
    print("\n" + "="*60)