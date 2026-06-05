def derivada_segunda_central_o4(func, x, h):
    numerador = -func(x + 2*h) + 16*func(x + h) - 30*func(x) + 16*func(x - h) - func(x - 2*h)
    denominador = 12 * (h**2)
    return numerador / denominador

def derivada_quarta_central(func, x, h=1e-3):
    numerador = func(x + 2*h) - 4*func(x + h) + 6*func(x) - 4*func(x - h) + func(x - 2*h)
    denominador = h**4
    return numerador / denominador