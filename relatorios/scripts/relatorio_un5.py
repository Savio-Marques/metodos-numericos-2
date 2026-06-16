import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../metodos')))
import numpy as np
import matplotlib.pyplot as plt
import pvc
import mef

def rodar_un5():
    print("="*50)
    print(" UNIDADE 5 - PROBLEMA DE VALOR DE CONTORNO (PVC)")
    print("="*50)
    
    # Parâmetros
    a, b = 0.0, 2.0
    u_a, u_b = 10.0, 1.0
    dx = 0.1
    
    # Diferenças Finitas
    P = lambda x: 7.0
    Q = lambda x: -1.0
    R = lambda x: 2.0
    
    x_df, u_df = pvc.resolver_pvc_diferencas_finitas(P, Q, R, a, b, u_a, u_b, dx)
    
    # Elementos Finitos (Bônus)
    N = int(np.round((b - a) / dx))
    B_mef = 7.0
    C_mef = -1.0
    D_mef = 2.0
    
    x_mef, u_mef = mef.resolver_mef_geral(B_mef, C_mef, D_mef, a, b, u_a, u_b, N)
    
    print("\nTabela de Resultados (x | Diferenças Finitas | MEF)")
    print("-" * 55)
    for i in range(len(x_df)):
        print(f"x = {x_df[i]:.1f} | DF: u(x) = {u_df[i]:.6f} | MEF: u(x) = {u_mef[i]:.6f}")
        
    # Plotar
    plt.figure(figsize=(8, 6))
    plt.plot(x_df, u_df, 'bo-', label='Diferenças Finitas')
    plt.plot(x_mef, u_mef, 'r*--', label='Elementos Finitos (Bônus)')
    plt.title("Solução do PVC: u'' + 7u' - u = 2")
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), "../resultados/grafico_pvc.png"))
    print("\nGráfico gerado: grafico_pvc.png")

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "../resultados/relatorio_un5_output.txt"), "w", encoding="utf-8") as f_out:
        import sys
        sys.stdout = f_out
        rodar_un5()
        sys.stdout = sys.__stdout__
    print("Resultados da Unidade 5 gerados com sucesso! Veja os arquivos na pasta resultados.")
