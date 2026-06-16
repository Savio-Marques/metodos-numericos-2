# Métodos Numéricos II - Trabalhos Práticos

Este repositório contém a implementação de diversos métodos numéricos desenvolvidos para a disciplina de **Métodos Numéricos II**, bem como os scripts automatizados para a geração dos relatórios e gráficos solicitados ao longo de 5 unidades.

## 📁 Estrutura do Projeto

O código foi organizado para separar claramente a lógica matemática (os métodos) da execução e geração de resultados (relatórios).

```text
MN2/
│
├── metodos/                  # Contém todas as implementações base dos métodos numéricos
│   ├── derivadas.py          # Diferenciação Numérica (O(h^4))
│   ├── imagens.py            # Filtros de Processamento de Imagens (Sobel, Laplace)
│   ├── integrais.py          # Integração Numérica (Newton-Cotes e Gauss)
│   ├── integrais_singulares.py # Integrais de Exponencial Simples e Dupla
│   ├── integrais_2d.py       # Integração Dupla (Gauss 2D)
│   ├── integrais_3d.py       # Integração Tripla (Gauss 3D)
│   ├── autovalores.py        # Métodos de Potência, Householder e QR
│   ├── pvi.py                # Problemas de Valor Inicial (Euler, RK4, Preditor-Corretor)
│   ├── pvc.py                # Problema de Valor de Contorno (Diferenças Finitas)
│   ├── mef.py                # Problema de Valor de Contorno (Elementos Finitos)
│   ├── svd.py                # Decomposição em Valores Singulares (SVD)
│   └── erros.py              # Cálculo de Erros Teóricos
│
├── relatorios/
│   ├── scripts/              # Scripts executáveis que chamam os métodos e rodam os testes
│   │   ├── relatorio_un1.py  # Gera os resultados da Unidade 1
│   │   ├── relatorio_un2.py  # Gera os resultados da Unidade 2
│   │   ├── relatorio_un3.py  # Gera os resultados da Unidade 3
│   │   ├── relatorio_un4.py  # Gera os resultados da Unidade 4
│   │   └── relatorio_un5.py  # Gera os resultados da Unidade 5
│   │
│   └── resultados/           # Pasta onde os txts, gráficos e imagens de saída são salvos
│
├── minha_foto.jpg            # Imagem de teste utilizada na Unidade 1
└── README.md                 # Este arquivo de documentação
```

---

## ⚙️ Pré-requisitos e Instalação

Para executar os scripts, você precisará do Python instalado e de algumas bibliotecas auxiliares para manipulação de matrizes e imagens.

Abra seu terminal ou prompt de comando e instale as dependências executando:
```bash
pip install numpy matplotlib pillow
```

---

## 🚀 Como Executar e Gerar os Relatórios

Você pode gerar os resultados (tabelas e gráficos) executando cada script da pasta `relatorios/scripts/` diretamente. 

Por exemplo, para gerar o relatório da **Unidade 1**, abra o terminal na raiz do projeto (`MN2/`) e rode:

```bash
python relatorios/scripts/relatorio_un1.py
```

### O que acontece quando você roda os scripts?
1. O script vai importar as funções matemáticas da pasta `metodos/`.
2. Vai realizar todos os testes de bancada que o professor pede em sala de aula (comparando com soluções exatas).
3. Vai salvar a saída formatada na pasta `relatorios/resultados/` em um arquivo de texto (ex: `relatorio_un1_output.txt`).
4. Gráficos gerados (se houver, como nas Unidades 1, 4 e 5) também serão salvos na pasta de resultados.

---

## 📝 Resumo das Unidades

- **Unidade 1:** Testes de Diferenciação Numérica e Processamento de Imagens (Sobel e Laplace).
- **Unidade 2:** Fórmulas de Newton-Cotes, Quadratura de Gauss, Exponenciais (Simples e Dupla) e Integrais Multidimensionais.
- **Unidade 3:** Autovalores e Autovetores. Métodos das Potências e Método Iterativo de QR acoplado com Matriz de Householder.
- **Unidade 4:** Solução de PVI. Métodos de Euler, Runge-Kutta de 4ª Ordem e Preditor-Corretor (Simulação de Queda Livre com atrito).
- **Unidade 5:** Solução de PVC com Método das Diferenças Finitas e bônus pelo Método dos Elementos Finitos (MEF).
