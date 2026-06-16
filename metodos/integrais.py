import numpy as np

def fechada_grau1(f, xi, xf):
    dx = xf - xi
    return (dx / 2.0) * (f(xi) + f(xf))

def fechada_grau2(f, xi, xf):
    # Regra de Simpson 1/3
    h = (xf - xi) / 2.0
    return (h / 3.0) * (f(xi) + 4*f(xi + h) + f(xf))

def fechada_grau3(f, xi, xf):
    # Regra de Simpson 3/8
    h = (xf - xi) / 3.0
    return (3.0 * h / 8.0) * (f(xi) + 3*f(xi + h) + 3*f(xi + 2*h) + f(xf))

def fechada_grau4(f, xi, xf):
    # Regra de Boole
    h = (xf - xi) / 4.0
    return (2.0 * h / 45.0) * (7*f(xi) + 32*f(xi + h) + 12*f(xi + 2*h) + 32*f(xi + 3*h) + 7*f(xf))


def aberta_grau1(f, xi, xf):
    # Regra do Trapézio Aberta
    h = (xf - xi) / 3.0
    return (3.0 * h / 2.0) * (f(xi + h) + f(xi + 2*h))

def aberta_grau2(f, xi, xf):
    # Fórmula de Milne
    h = (xf - xi) / 4.0
    return (4.0 * h / 3.0) * (2*f(xi + h) - f(xi + 2*h) + 2*f(xi + 3*h))

def aberta_grau3(f, xi, xf):
    # Grau 3 Aberta
    h = (xf - xi) / 5.0
    return (5.0 * h / 24.0) * (11*f(xi + h) + f(xi + 2*h) + f(xi + 3*h) + 11*f(xi + 4*h))

def aberta_grau4(f, xi, xf):
    # Grau 4 Aberta
    h = (xf - xi) / 6.0
    return (3.0 * h / 10.0) * (11*f(xi + h) - 14*f(xi + 2*h) + 26*f(xi + 3*h) - 14*f(xi + 4*h) + 11*f(xi + 5*h))

def gauss_legendre_2(f, xi, xf):
    # Raízes e Pesos para n=2
    alpha = [-np.sqrt(1/3), np.sqrt(1/3)]
    w = [1.0, 1.0]
    
    # Transformação de domínio
    meio = (xi + xf) / 2.0
    metade_dx = (xf - xi) / 2.0
    
    integral = 0.0
    for k in range(2):
        x_k = meio + metade_dx * alpha[k]
        integral += w[k] * f(x_k)
        
    return metade_dx * integral

def gauss_legendre_3(f, xi, xf):
    # Raízes e Pesos para n=3
    alpha = [-np.sqrt(3/5), 0.0, np.sqrt(3/5)]
    w = [5/9, 8/9, 5/9]
    
    meio = (xi + xf) / 2.0
    metade_dx = (xf - xi) / 2.0
    
    integral = 0.0
    for k in range(3):
        x_k = meio + metade_dx * alpha[k]
        integral += w[k] * f(x_k)
        
    return metade_dx * integral

def gauss_legendre_4(f, xi, xf):
    # Raízes e Pesos para n=4
    termo = (2/7) * np.sqrt(6/5)
    alpha = [
        -np.sqrt(3/7 + termo),
        -np.sqrt(3/7 - termo),
         np.sqrt(3/7 - termo),
         np.sqrt(3/7 + termo)
    ]
    
    w1_4 = (18 - np.sqrt(30)) / 36
    w2_3 = (18 + np.sqrt(30)) / 36
    w = [w1_4, w2_3, w2_3, w1_4]
    
    meio = (xi + xf) / 2.0
    metade_dx = (xf - xi) / 2.0
    
    integral = 0.0
    for k in range(4):
        x_k = meio + metade_dx * alpha[k]
        integral += w[k] * f(x_k)
        
    return metade_dx * integral

def gauss_hermite(f, n):
    tabela = {
        2: {'x': [-1/np.sqrt(2), 1/np.sqrt(2)], 
            'w': [np.sqrt(np.pi)/2, np.sqrt(np.pi)/2]},
        3: {'x': [-np.sqrt(3/2), 0.0, np.sqrt(3/2)], 
            'w': [np.sqrt(np.pi)/6, 2*np.sqrt(np.pi)/3, np.sqrt(np.pi)/6]},
        4: {'x': [-1.65068012, -0.52464762, 0.52464762, 1.65068012], 
            'w': [0.08131283, 0.80491409, 0.80491409, 0.08131283]}
    }
    
    x_k = tabela[n]['x']
    w_k = tabela[n]['w']
    
    return sum(w * f(x) for w, x in zip(w_k, x_k))


def gauss_laguerre(f, n):
    tabela = {
        2: {'x': [2 - np.sqrt(2), 2 + np.sqrt(2)], 
            'w': [(2 + np.sqrt(2))/4, (2 - np.sqrt(2))/4]},
        3: {'x': [0.41577455, 2.29428036, 6.28994508], 
            'w': [0.71109300, 0.27851773, 0.01038925]},
        4: {'x': [0.32254768, 1.74576110, 4.53662029, 9.39507091], 
            'w': [0.60315410, 0.35741869, 0.03888790, 0.00053929]}
    }
    
    x_k = tabela[n]['x']
    w_k = tabela[n]['w']
    
    return sum(w * f(x) for w, x in zip(w_k, x_k))


def gauss_chebyshev(f, n):
    integral = 0.0
    peso = np.pi / n
    for k in range(1, n + 1):
        x_k = np.cos(((2 * k - 1) / (2 * n)) * np.pi)
        integral += peso * f(x_k)
        
    return integral