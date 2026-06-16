import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../metodos')))
import numpy as np
import pvi
import matplotlib.pyplot as plt

def integrador_rk4(F, S0, t0, dt, num_passos):
    t = t0
    S = np.array(S0, dtype=float)
    historico_t = [t]
    historico_S = [S]
    for _ in range(num_passos):
        S = pvi.runge_kutta_4(F, S, t, dt)
        t += dt
        historico_t.append(t)
        historico_S.append(S)
    return np.array(historico_t), np.array(historico_S)

def F_pvi1(S, t):
    # S é um escalar y(t), mas tratado como 1D array ou escalar
    return (2.0/3.0) * S

def F_pvi2(S, t):
    # S = [v, y]
    v = S[0]
    y = S[1]
    g = 10.0
    k = 0.25
    m = 2.0
    dv_dt = -g - (k/m)*v
    dy_dt = v
    return np.array([dv_dt, dy_dt])

def rodar_pvi1():
    print("="*50)
    print(" UNIDADE 4 - PVI-1: y'(t) = (2/3)y(t), y(0) = 2")
    print("="*50)
    
    t0 = 0.0
    S0 = 2.0
    dt = 0.5
    t_max = 2.5
    num_passos = int(np.round((t_max - t0) / dt))

    print("\n--- Euler Explícito ---")
    t_ee, S_ee = pvi.euler_explicito(F_pvi1, S0, t0, dt, num_passos)
    for i in range(len(t_ee)):
        print(f"t = {t_ee[i]:.2f} | y = {S_ee[i]:.6f}")

    print("\n--- Euler Implícito ---")
    t_ei, S_ei = pvi.euler_implicito(F_pvi1, S0, t0, dt, num_passos)
    for i in range(len(t_ei)):
        print(f"t = {t_ei[i]:.2f} | y = {S_ei[i]:.6f}")

    print("\n--- Runge-Kutta de 4ª Ordem ---")
    t_rk, S_rk = integrador_rk4(F_pvi1, S0, t0, dt, num_passos)
    for i in range(len(t_rk)):
        print(f"t = {t_rk[i]:.2f} | y = {S_rk[i]:.6f}")

    print("\n--- Preditor-Corretor 4ª Ordem ---")
    def cond_parada1(t, S):
        return t >= t_max - 1e-9
    t_pc, S_pc = pvi.preditor_corretor_4(F_pvi1, S0, t0, dt, cond_parada1)
    for i in range(len(t_pc)):
        print(f"t = {t_pc[i]:.2f} | y = {S_pc[i]:.6f}")

def rodar_pvi2():
    print("\n" + "="*50)
    print(" UNIDADE 4 - PVI-2: Queda Livre com Atrito")
    print(" v(0)=5, y(0)=200, g=10, k=0.25, m=2")
    print("="*50)
    
    t0 = 0.0
    S0 = np.array([5.0, 200.0]) # v0, y0
    dt = 0.5
    # Condição de parada geral: y <= 0 (chegou no chão)
    # Para Euler e RK, vamos rodar por um tempo fixo de 20s para garantir
    t_max = 20.0
    num_passos = int(np.round((t_max - t0) / dt))

    print("\n--- Runge-Kutta de 4ª Ordem ---")
    t_rk, S_rk = integrador_rk4(F_pvi2, S0, t0, dt, num_passos)
    for i in range(0, len(t_rk), 4): # imprimir a cada 4 passos (2s)
        print(f"t = {t_rk[i]:.2f}s | v = {S_rk[i,0]:.4f} m/s | y = {S_rk[i,1]:.4f} m")

    print("\n--- Preditor-Corretor 4ª Ordem (Para quando atingir o mar y <= 0) ---")
    def cond_parada2(t, S):
        return S[1] <= 0.0
    t_pc, S_pc = pvi.preditor_corretor_4(F_pvi2, S0, t0, dt, cond_parada2)
    for i in range(0, len(t_pc), 4):
        print(f"t = {t_pc[i]:.2f}s | v = {S_pc[i,0]:.4f} m/s | y = {S_pc[i,1]:.4f} m")
    
    # Imprime último ponto exato (impacto)
    print(f"IMPACTO NO MAR: t = {t_pc[-1]:.2f}s | v = {S_pc[-1,0]:.4f} m/s | y = {S_pc[-1,1]:.4f} m")

    # Gráfico do PVI-2
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_pc, S_pc[:, 1], 'b-', label='Altura y(t)')
    plt.title('PVI-2: Altura ao longo do tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Altura (m)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_pc, S_pc[:, 0], 'r-', label='Velocidade v(t)')
    plt.title('PVI-2: Velocidade ao longo do tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocidade (m/s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "../resultados/grafico_pvi2.png"))
    print("\nGráfico gerado: grafico_pvi2.png")


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "../resultados/relatorio_un4_output.txt"), "w", encoding="utf-8") as f_out:
        import sys
        sys.stdout = f_out
        rodar_pvi1()
        rodar_pvi2()
        sys.stdout = sys.__stdout__
    print("Resultados da Unidade 4 gerados com sucesso! Veja os arquivos na pasta resultados.")
