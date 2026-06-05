import numpy as np
import matplotlib.pyplot as plt
import imagens as img_proc
from PIL import Image
import os

if __name__ == "__main__":
    print("Carregando imagem local")
    
    nome_ficheiro = "minha_foto.jpg"
    
    if not os.path.exists(nome_ficheiro):
        print(f"Erro: Não foi possível encontrar o ficheiro '{nome_ficheiro}'!")
        print("Certifique-se de que a imagem está na mesma pasta que este script main.py.")
    else:
        img_pil = Image.open(nome_ficheiro)

        img_pil = img_pil.convert('L')
        
        img_pil = img_pil.resize((200, 200))
        
        imagem_real = np.array(img_pil) / 255.0
        
        resultado_sobel = img_proc.algoritmo_1_sobel(imagem_real, threshold=0.15)
        
        resultado_laplace = img_proc.algoritmo_2_laplace(imagem_real, tolerancia=0.04)
        
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        axs[0].imshow(imagem_real, cmap='gray')
        axs[0].set_title('Imagem Original')
        axs[0].axis('off')
        
        axs[1].imshow(resultado_sobel, cmap='gray')
        axs[1].set_title('Sobel (Bordas)')
        axs[1].axis('off')
        
        axs[2].imshow(resultado_laplace, cmap='gray')
        axs[2].set_title('Laplace (Bordas)')
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        print("Processamento concluído")