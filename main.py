import numpy as np
import pvc

def solucao_exata_pvc1(x):
    """ Equação (4) da Aula #27 """
    return (np.exp(-x) - np.exp(x)) / (np.exp(-1) - np.exp(1))

if __name__ == "__main__":
    print("="*70)
    print(" TAREFA 19: PROBLEMAS DE VALOR DE CONTORNO (DIFERENÇAS FINITAS)")
    print("="*70)
    
    print("\n1) RESOLUÇÃO DO PVC1 (1D) COM N = 8")
    print("-" * 70)
    
    x_n8, y_num_n8 = pvc.resolver_pvc1_1d(N=8)
    y_exato = solucao_exata_pvc1(x_n8)
    
    print(f"{'Nó (i)':<8} | {'x':<6} | {'y Numérico (N=8)':<20} | {'y Exato':<20} | {'Erro Relativo (%)':<15}")
    print("-" * 70)
    
    for i in range(len(x_n8)):
        if y_exato[i] != 0:
            erro_rel = abs((y_exato[i] - y_num_n8[i]) / y_exato[i]) * 100
            erro_str = f"{erro_rel:.4f} %"
        else:
            erro_str = "0.0000 % (Borda)"
            
        print(f"{i:<8} | {x_n8[i]:<6.3f} | {y_num_n8[i]:<20.6f} | {y_exato[i]:<20.6f} | {erro_str:<15}")

    print("\n\n2) RESOLUÇÃO DO PVC2 (2D) COM N = 8")
    print("-" * 70)
    
    U_n8 = pvc.resolver_pvc2_2d(N=8, f_val=4.0)
    
    valor_centro_n4 = -0.28125
    valor_centro_n8 = U_n8[4, 4]
    
    erro_relativo_centro = abs((valor_centro_n4 - valor_centro_n8) / valor_centro_n4) * 100
    
    print("Comparação no Centro da Placa (x = 0.5, y = 0.5):")
    print(f" -> Valor aproximado com N=4 (Aula 27) : {valor_centro_n4:.6f}")
    print(f" -> Valor aproximado com N=8 (Nosso)   : {valor_centro_n8:.6f}")
    print(f" -> Discrepância / Mudança             : {erro_relativo_centro:.4f} %")
    
    print("\nNota: Ao duplicar a resolução (de N=4 para N=8), o número de equações a")
    print("resolver simultaneamente saltou de 9 para 49. O nosso motor resolveu o")
    print("sistema linear 49x49 de forma instantânea!")
    
    print("\n" + "="*70)