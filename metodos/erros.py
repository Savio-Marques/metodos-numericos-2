from derivadas import derivada_quarta_central

def erro_teorico_milne(f, a, b):
    dx = b - a
    x_barra = (a + b) / 2.0
    
    f_quatro_linhas = derivada_quarta_central(f, x_barra)
    
    return (7.0 / 23040.0) * (dx**5) * f_quatro_linhas