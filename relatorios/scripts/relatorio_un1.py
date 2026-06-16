import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../metodos')))
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import derivadas as der
import imagens as img
import os

def f(x):
    return np.sin(x)

def f_duas_linhas_exata(x):
    return -np.sin(x)

def f_quatro_linhas_exata(x):
    return np.sin(x)

def rodar_testes_derivadas():
    print("="*50)
    print(" UNIDADE 1 - DIFERENCIAÇÃO NUMÉRICA")
    print("="*50)
    x = np.pi / 4
    h_vals = [1e-1, 1e-2, 1e-3, 1e-4]
    
    print("\n--- Derivada Segunda Central O(h^4) de f(x) = sin(x) em x = pi/4 ---")
    exata2 = f_duas_linhas_exata(x)
    print(f"Valor Exato: {exata2:.8f}")
    print(f"{'h':<10} | {'Aproximado':<12} | {'Erro Absoluto':<15}")
    print("-" * 43)
    for h in h_vals:
        aprox2 = der.derivada_segunda_central_o4(f, x, h)
        erro2 = abs(aprox2 - exata2)
        print(f"{h:<10} | {aprox2:<12.8f} | {erro2:<15.2e}")

    print("\n--- Derivada Quarta Central de f(x) = sin(x) em x = pi/4 ---")
    exata4 = f_quatro_linhas_exata(x)
    print(f"Valor Exato: {exata4:.8f}")
    print(f"{'h':<10} | {'Aproximado':<12} | {'Erro Absoluto':<15}")
    print("-" * 43)
    for h in h_vals:
        aprox4 = der.derivada_quarta_central(f, x, h)
        erro4 = abs(aprox4 - exata4)
        print(f"{h:<10} | {aprox4:<12.8f} | {erro4:<15.2e}")

def rodar_processamento_imagens():
    print("\n" + "="*50)
    print(" UNIDADE 1 - PROCESSAMENTO DE IMAGENS")
    print("="*50)
    
    arquivo_imagem = os.path.join(os.path.dirname(__file__), "../../minha_foto.jpg")
    if not os.path.exists(arquivo_imagem):
        print(f"Erro: O arquivo '{arquivo_imagem}' não foi encontrado.")
        return

    print("Carregando imagem...")
    img_pil = Image.open(arquivo_imagem).convert('L') # Convertendo para tons de cinza
    img_array = np.array(img_pil, dtype=np.float32) / 255.0 # Normalizando para [0, 1]
    
    print("Aplicando Filtro de Sobel (pode levar alguns segundos)...")
    img_sobel = img.algoritmo_1_sobel(img_array, threshold=0.3)
    
    print("Aplicando Filtro de Laplace (pode levar alguns segundos)...")
    img_laplace = img.algoritmo_2_laplace(img_array, tolerancia=0.03)
    
    print("Salvando resultados...")
    plt.imsave(os.path.join(os.path.dirname(__file__), "../resultados/resultado_sobel.jpg"), img_sobel, cmap='gray')
    plt.imsave(os.path.join(os.path.dirname(__file__), "../resultados/resultado_laplace.jpg"), img_laplace, cmap='gray')
    
    # Criar imagem com as 3 enfileiradas
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original (Tons de Cinza)')
    axes[0].axis('off')
    
    axes[1].imshow(img_sobel, cmap='gray')
    axes[1].set_title('Filtro de Sobel')
    axes[1].axis('off')
    
    axes[2].imshow(img_laplace, cmap='gray')
    axes[2].set_title('Filtro de Laplace')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "../resultados/resultado_3_imagens_enfileiradas.png"), dpi=300)
    plt.close(fig)
    
    print("Processamento concluído. Imagens salvas como 'resultado_sobel.jpg', 'resultado_laplace.jpg' e 'resultado_3_imagens_enfileiradas.png'.")

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "../resultados/relatorio_un1_output.txt"), "w", encoding="utf-8") as f_out:
        import sys
        sys.stdout = f_out
        rodar_testes_derivadas()
        rodar_processamento_imagens()
        sys.stdout = sys.__stdout__
    print("Resultados da Unidade 1 gerados com sucesso! Veja os arquivos na pasta resultados.")
