import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../metodos')))
import numpy as np
import autovalores as av

def matriz_simetrica_exemplo():
    # Matriz simétrica definida positiva
    return np.array([
        [4.0, -1.0,  1.0],
        [-1.0, 3.0, -2.0],
        [1.0, -2.0,  3.0]
    ])

def rodar_metodos_potencia():
    print("="*50)
    print(" UNIDADE 3 - MÉTODOS DAS POTÊNCIAS")
    print("="*50)
    
    A = matriz_simetrica_exemplo()
    v0 = np.array([1.0, 1.0, 1.0])
    
    print("Matriz A:")
    print(A)
    print("\nVetor inicial v0:", v0)
    
    # Exato para comparar
    autovals_exatos, autovecs_exatos = np.linalg.eigh(A)
    print("\nAutovalores exatos (numpy):", autovals_exatos)
    
    # Regular
    lamb_reg, v_reg, iter_reg = av.potencia_regular(A, v0)
    print(f"\n--- Método da Potência Regular ---")
    print(f"Autovalor dominante: {lamb_reg:.6f}")
    print(f"Autovetor correspondente: {v_reg}")
    print(f"Iterações: {iter_reg}")

    # Inverso
    lamb_inv, v_inv, iter_inv = av.potencia_inverso(A, v0)
    print(f"\n--- Método da Potência Inverso ---")
    print(f"Autovalor menor (em módulo): {lamb_inv:.6f}")
    print(f"Autovetor correspondente: {v_inv}")
    print(f"Iterações: {iter_inv}")

    # Deslocamento
    mu = 2.5
    lamb_desl, v_desl, iter_desl = av.potencia_deslocamento(A, v0, mu)
    print(f"\n--- Método da Potência com Deslocamento (mu = {mu}) ---")
    print(f"Autovalor mais próximo de {mu}: {lamb_desl:.6f}")
    print(f"Autovetor correspondente: {v_desl}")
    print(f"Iterações: {iter_desl}")

def rodar_householder_qr():
    print("\n" + "="*50)
    print(" UNIDADE 3 - HOUSEHOLDER + QR (Caixas Pretas)")
    print("="*50)

    A = matriz_simetrica_exemplo()
    print("Matriz Original A:")
    print(A)
    
    print("\n--- Passo 1: Redução Tridiagonal por Householder ---")
    A_tridiag = av.householder_tridiagonal(A)
    print("Matriz Tridiagonal Simétrica (H * A * H):")
    # Zera valores muito pequenos para visualização
    A_tridiag[np.abs(A_tridiag) < 1e-10] = 0.0
    print(A_tridiag)
    
    print("\n--- Passo 2: Método QR iterativo ---")
    autovalores, iteracoes = av.metodo_qr(A_tridiag)
    print("Autovalores encontrados na diagonal principal:")
    print(autovalores)
    print(f"Número de iterações do método QR: {iteracoes}")

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "../resultados/relatorio_un3_output.txt"), "w", encoding="utf-8") as f_out:
        import sys
        sys.stdout = f_out
        rodar_metodos_potencia()
        rodar_householder_qr()
        sys.stdout = sys.__stdout__
    print("Resultados da Unidade 3 gerados com sucesso! Veja os arquivos na pasta resultados.")
