import numpy as np

def convolucao_2d(imagem, kernel):

    k_alt, k_larg = kernel.shape
    pad_h = k_alt // 2
    pad_w = k_larg // 2
    
    imagem_pad = np.pad(imagem, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    resultado = np.zeros_like(imagem)
    
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            regiao = imagem_pad[i:i+k_alt, j:j+k_larg]
            resultado[i, j] = np.sum(regiao * kernel)
            
    return resultado

def algoritmo_1_sobel(imagem, threshold=0.5):

    filtro_gaussiano = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16.0
    img_suavizada = convolucao_2d(imagem, filtro_gaussiano)
    
    sobel_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ])
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    
    A = convolucao_2d(img_suavizada, sobel_x)
    B = convolucao_2d(img_suavizada, sobel_y)
    
    C = np.sqrt(A**2 + B**2)
    
    D = np.zeros_like(C)
    D[C > threshold] = 1.0 
    
    imagem_final = 1.0 - D
    
    return imagem_final


def algoritmo_2_laplace(imagem, tolerancia=0.01):
    filtro_gaussiano = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16.0
    img_suavizada = convolucao_2d(imagem, filtro_gaussiano)
    
    filtro_laplace = np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ])
    A = convolucao_2d(img_suavizada, filtro_laplace)
    
    B = np.zeros_like(A)
    B[np.abs(A) > tolerancia] = 1.0
    
    imagem_final = 1.0 - B
    
    return imagem_final