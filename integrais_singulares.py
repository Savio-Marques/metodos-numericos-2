import numpy as np
import integrais as intg

def exp_simples(f, a, b, s):
    x_s = (a + b) / 2.0 + ((b - a) / 2.0) * np.tanh(s)
    dx_ds = ((b - a) / 2.0) * (1.0 / np.cosh(s)**2)
    return f(x_s) * dx_ds

def exp_dupla(f, a, b, s):
    termo_sinh = np.sinh(s)
    termo_cosh = np.cosh(s)
    
    arg_cosh = (np.pi / 2.0) * termo_sinh
    
    if np.abs(arg_cosh) > 700: 
        return 0.0 
        
    x_s = (a + b) / 2.0 + ((b - a) / 2.0) * np.tanh(arg_cosh)
    dx_ds = ((b - a) / 2.0) * ((np.pi / 2.0) * termo_cosh / np.cosh(arg_cosh)**2)
    return f(x_s) * dx_ds

def calcular_integral_composta(metodo, f, a, b, tolerancia=1e-6):
    N = 1
    integral_anterior = None
    
    while True:
        dx = (b - a) / N
        integral_atual = sum(metodo(f, a + i*dx, a + (i+1)*dx) for i in range(N))
            
        if integral_anterior is not None:
            if integral_atual != 0:
                erro_relativo = abs((integral_atual - integral_anterior) / integral_atual)
            else:
                erro_relativo = 0
                
            if erro_relativo < tolerancia:
                return integral_atual
                
        integral_anterior = integral_atual
        N *= 2

def integral_singularidade(f, a, b, tipo='simples', tol=1e-5):
    c = 1.0
    integral_anterior = None
    
    while True:
        if tipo == 'simples':
            f_barra = lambda s: exp_simples(f, a, b, s)
        else:
            f_barra = lambda s: exp_dupla(f, a, b, s)
            
        # Resolve usando o nosso método de Gauss-Legendre de 4 pontos (o mais rápido)
        integral_atual = calcular_integral_composta(intg.gauss_legendre_4, f_barra, -c, c, tol)
        
        # Analisa se aumentar o 'c' ainda faz alguma diferença no resultado
        if integral_anterior is not None:
            if integral_atual != 0:
                erro_relativo = abs((integral_atual - integral_anterior) / integral_atual)
            else:
                erro_relativo = 0
                
            if erro_relativo < tol:
                return integral_atual, c
                
        integral_anterior = integral_atual
        
        # Para a dupla, o professor avisa que não podemos passar muito de c=3.5
        if tipo == 'dupla' and c >= 3.5:
            return integral_atual, c
            
        c += 0.5