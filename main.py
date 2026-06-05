import numpy as np
import mef

def solucao_exata(x):
    return (np.exp(-x) - np.exp(x)) / (np.exp(-1) - np.exp(1))

if __name__ == "__main__":
    print("="*60)
    print(" TAREFA DE ELEMENTOS FINITOS (MEF) - AULA 28")
    print("="*60)
    
    N = 8
    x, y_mef = mef.resolver_mef_1d(N)
    y_exato = solucao_exata(x)
    
    print(f"{'Nó':<5} | {'x':<6} | {'y MEF':<15} | {'y Exato':<15}")
    print("-" * 55)
    for i in range(len(x)):
        print(f"{i:<5} | {x[i]:<6.3f} | {y_mef[i]:<15.6f} | {y_exato[i]:<15.6f}")
    
    print("="*60)