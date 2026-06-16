import numpy as np

def jacobi_simetrica(A_sym, tol=1e-9, max_iter=1000):
    n = A_sym.shape[0]
    A_k = np.copy(A_sym)
    V = np.eye(n)
    
    for _ in range(max_iter):
        off_diag = np.triu(np.abs(A_k), k=1)
        if np.max(off_diag) < tol:
            break
        p, q = np.unravel_index(np.argmax(off_diag), A_k.shape)
        
        if A_k[p, p] == A_k[q, q]:
            theta = np.pi / 4.0
        else:
            theta = 0.5 * np.arctan2(2.0 * A_k[p, q], A_k[p, p] - A_k[q, q])
            
        c = np.cos(theta)
        s = np.sin(theta)
        
        J = np.eye(n)
        J[p, p] = c
        J[q, q] = c
        J[p, q] = s
        J[q, p] = -s
        
        A_k = np.dot(J.T, np.dot(A_k, J))
        V = np.dot(V, J)
        
    autovalores = np.diag(A_k)
    
    idx = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx]
    V = V[:, idx]
    
    return autovalores, V

def completar_base_ortogonal(Matriz, r):
    m = Matriz.shape[0]
    col_idx = r
    for i in range(m):
        if col_idx >= m: break
        
        v = np.zeros(m)
        v[i] = 1.0
        
        for j in range(col_idx):
            v -= np.dot(Matriz[:, j], v) * Matriz[:, j]
            
        norma = np.linalg.norm(v)
        if norma > 1e-6:
            Matriz[:, col_idx] = v / norma
            col_idx += 1
            
    return Matriz

def executar_svd():
    print("="*60)
    print(" DECOMPOSIÇÃO EM VALORES SINGULARES (SVD)")
    print("="*60)
    
    try:
        m = int(input("Digite o número de linhas (m): "))
        n = int(input("Digite o número de colunas (n): "))
    except ValueError:
        print("Erro: Deve digitar números inteiros!")
        return

    A = np.zeros((m, n))
    print(f"\nDigite os elementos da matriz {m}x{n} (separados por espaço):")
    for i in range(m):
        linha = list(map(float, input(f"Linha {i+1}: ").split()))
        if len(linha) != n:
            print(f"Erro: A linha deve conter exatamente {n} números!")
            return
        A[i, :] = linha
        
    print("\n--- INICIANDO CÁLCULOS SVD ---")
    
    if m >= n:
        A_bar = np.dot(A.T, A)
        print(f"-> Caso m >= n detectado. Calculando A_bar = A^T . A (Tamanho {n}x{n})")
    else:
        A_bar = np.dot(A, A.T)
        print(f"-> Caso m < n detectado. Calculando A_bar = A . A^T (Tamanho {m}x{m})")
        
    autovalores, autovetores = jacobi_simetrica(A_bar)
    autovalores[autovalores < 1e-12] = 0.0 
    
    sigma = np.sqrt(autovalores)
    
    posto = np.sum(sigma > 1e-7)
    print(f"-> Posto da matriz A (Rank): {posto}")
    
    Sigma_mat = np.zeros((m, n))
    U = np.zeros((m, m))
    V = np.zeros((n, n))
    
    if m >= n:
        V = autovetores
        for i in range(n):
            if i < posto:
                Sigma_mat[i, i] = sigma[i]
                U[:, i] = np.dot(A, V[:, i]) / sigma[i]
        U = completar_base_ortogonal(U, posto)
    else:
        U = autovetores
        for i in range(m):
            if i < posto:
                Sigma_mat[i, i] = sigma[i]
                V[:, i] = np.dot(A.T, U[:, i]) / sigma[i]
        V = completar_base_ortogonal(V, posto)
        
    print("\n[RESULTADOS FINAIS]")
    print("\nMatriz U:")
    print(np.round(U, 4))
    
    print("\nMatriz Sigma:")
    print(np.round(Sigma_mat, 4))
    
    print("\nMatriz V^T:")
    print(np.round(V.T, 4))
    
    A_reconstruida = np.dot(U, np.dot(Sigma_mat, V.T))
    erro_reconstrucao = np.linalg.norm(A - A_reconstruida)
    
    print("\n" + "="*60)
    print(" VERIFICAÇÃO FINAL: U . Sigma . V^T == A")
    print("="*60)
    print("Matriz Original A:")
    print(A)
    print("\nMatriz Reconstruída (U . Sigma . V^T):")
    print(np.round(A_reconstruida, 4))
    
    if erro_reconstrucao < 1e-5:
        print(f"\n[SUCESSO] (Erro: {erro_reconstrucao:.2e})")
    else:
        print(f"\n[FALHA] (Erro: {erro_reconstrucao:.2e})")

if __name__ == "__main__":
    executar_svd()