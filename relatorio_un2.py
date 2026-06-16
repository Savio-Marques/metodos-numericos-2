import numpy as np
import integrais as intg
import integrais_singulares as ising
import integrais_2d as i2d
import integrais_3d as i3d

def rodar_newton_cotes():
    print("\n" + "="*50)
    print(" 1) NEWTON-COTES (Abertas e Fechadas)")
    print("="*50)
    
    # Testaremos f(x) = e^x no intervalo [0, 1]
    f = lambda x: np.exp(x)
    a, b = 0.0, 1.0
    exato = np.exp(1) - 1.0
    
    print(f"Função: f(x) = e^x no intervalo [{a}, {b}]")
    print(f"Valor Exato: {exato:.6f}\n")
    
    # Fórmulas Fechadas
    print("Fórmulas Fechadas:")
    res_f1 = intg.fechada_grau1(f, a, b)
    res_f2 = intg.fechada_grau2(f, a, b)
    res_f3 = intg.fechada_grau3(f, a, b)
    res_f4 = intg.fechada_grau4(f, a, b)
    print(f"Grau 1 (Trapézio): {res_f1:.6f} | Erro: {abs(res_f1 - exato):.2e}")
    print(f"Grau 2 (Simpson 1/3): {res_f2:.6f} | Erro: {abs(res_f2 - exato):.2e}")
    print(f"Grau 3 (Simpson 3/8): {res_f3:.6f} | Erro: {abs(res_f3 - exato):.2e}")
    print(f"Grau 4 (Boole):       {res_f4:.6f} | Erro: {abs(res_f4 - exato):.2e}")

    # Fórmulas Abertas
    print("\nFórmulas Abertas:")
    res_a1 = intg.aberta_grau1(f, a, b)
    res_a2 = intg.aberta_grau2(f, a, b)
    res_a3 = intg.aberta_grau3(f, a, b)
    res_a4 = intg.aberta_grau4(f, a, b)
    print(f"Grau 1: {res_a1:.6f} | Erro: {abs(res_a1 - exato):.2e}")
    print(f"Grau 2 (Milne): {res_a2:.6f} | Erro: {abs(res_a2 - exato):.2e}")
    print(f"Grau 3: {res_a3:.6f} | Erro: {abs(res_a3 - exato):.2e}")
    print(f"Grau 4: {res_a4:.6f} | Erro: {abs(res_a4 - exato):.2e}")


def rodar_gauss():
    print("\n" + "="*50)
    print(" 2) FÓRMULAS DE GAUSS")
    print("="*50)

    # Legendre
    f = lambda x: np.exp(x)
    a, b = 0.0, 1.0
    exato = np.exp(1) - 1.0
    print("--- Gauss-Legendre (f(x)=e^x em [0,1]) ---")
    for n in [2, 3, 4]:
        func_gl = getattr(intg, f'gauss_legendre_{n}')
        res = func_gl(f, a, b)
        print(f"n = {n}: {res:.6f} | Erro: {abs(res - exato):.2e}")

    # Hermite
    print("\n--- Gauss-Hermite (w(x)=e^-x^2) ---")
    f_h = lambda x: x**2
    exato_h = np.sqrt(np.pi) / 2.0
    print(f"Função f(x) = x^2, Valor Exato: {exato_h:.6f}")
    for n in [2, 3, 4]:
        res = intg.gauss_hermite(f_h, n)
        print(f"n = {n}: {res:.6f} | Erro: {abs(res - exato_h):.2e}")

    # Laguerre
    print("\n--- Gauss-Laguerre (w(x)=e^-x) ---")
    f_l = lambda x: x**3
    exato_l = 6.0 # 3!
    print(f"Função f(x) = x^3, Valor Exato: {exato_l:.6f}")
    for n in [2, 3, 4]:
        res = intg.gauss_laguerre(f_l, n)
        print(f"n = {n}: {res:.6f} | Erro: {abs(res - exato_l):.2e}")

    # Chebyshev
    print("\n--- Gauss-Chebyshev (w(x)=1/sqrt(1-x^2)) ---")
    f_c = lambda x: x**2
    exato_c = np.pi / 2.0
    print(f"Função f(x) = x^2, Valor Exato: {exato_c:.6f}")
    for n in [2, 3, 4]:
        res = intg.gauss_chebyshev(f_c, n)
        print(f"n = {n}: {res:.6f} | Erro: {abs(res - exato_c):.2e}")

def rodar_exponenciais():
    print("\n" + "="*50)
    print(" 3) EXPONENCIAL SIMPLES E DUPLA")
    print("="*50)
    
    # Testaremos f(x) = 1/sqrt(x) em [0, 1], integral exata = 2.0
    f_sing = lambda x: 1.0 / np.sqrt(x) if x > 0 else 0
    a, b = 0.0, 1.0
    exato = 2.0
    print(f"Função singular f(x) = 1/sqrt(x) no intervalo [{a}, {b}]")
    print(f"Valor Exato: {exato:.6f}\n")

    res_simples, c_simp = ising.integral_singularidade(f_sing, a, b, tipo='simples')
    erro_simples = abs(res_simples - exato)
    print(f"Exponencial Simples: {res_simples:.6f} | Limite c={c_simp} | Erro: {erro_simples:.2e}")

    res_dupla, c_dup = ising.integral_singularidade(f_sing, a, b, tipo='dupla')
    erro_dupla = abs(res_dupla - exato)
    print(f"Exponencial Dupla  : {res_dupla:.6f} | Limite c={c_dup} | Erro: {erro_dupla:.2e}")

def rodar_integrais_multidimensionais():
    print("\n" + "="*50)
    print(" 4) INTEGRAIS DE ÁREAS E VOLUMES")
    print("="*50)

    # 2D: f(u, v) = u^2 + v^2 em [-1, 1]x[-1, 1]
    f2d = lambda u, v: u**2 + v**2
    exato2d = 8.0 / 3.0
    print("--- Integral 2D (Quadratura de Gauss 3 pontos) ---")
    print(f"Função f(u,v) = u^2 + v^2 no domínio [-1, 1]x[-1, 1]")
    res_2d = i2d.gauss_legendre_2d_3pontos(f2d)
    print(f"Aproximado: {res_2d:.6f} | Exato: {exato2d:.6f} | Erro: {abs(res_2d - exato2d):.2e}")

    # 3D: f(u, v, w) = u^2 + v^2 + w^2 em [-1, 1]x[-1, 1]x[-1, 1]
    f3d = lambda u, v, w: u**2 + v**2 + w**2
    exato3d = 8.0
    print("\n--- Integral 3D (Quadratura de Gauss 3 pontos) ---")
    print(f"Função f(u,v,w) = u^2 + v^2 + w^2 no domínio [-1, 1]x[-1, 1]x[-1, 1]")
    res_3d = i3d.gauss_legendre_3d_3pontos(f3d)
    print(f"Aproximado: {res_3d:.6f} | Exato: {exato3d:.6f} | Erro: {abs(res_3d - exato3d):.2e}")

if __name__ == "__main__":
    with open("relatorio_un2_output.txt", "w", encoding="utf-8") as f_out:
        import sys
        sys.stdout = f_out
        rodar_newton_cotes()
        rodar_gauss()
        rodar_exponenciais()
        rodar_integrais_multidimensionais()
        sys.stdout = sys.__stdout__
    print("Resultados da Unidade 2 gerados com sucesso! Veja 'relatorio_un2_output.txt'.")
