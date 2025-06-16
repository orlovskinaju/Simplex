import re
import random
import numpy as np
from itertools import combinations

def gerenciar_bases_aleatorias(num_colunas, num_restricoes, max_tentativas=None):
    if num_colunas < num_restricoes:
        raise ValueError("Número de colunas deve ser ≥ número de restrições")
    
    # Gera todas combinações possíveis
    todas_bases = list(combinations(range(num_colunas), num_restricoes))
    random.shuffle(todas_bases)  # Embaralha para aleatoriedade
    
    # Controla o número máximo de tentativas
    tentativas = 0
    max_tentativas = max_tentativas or len(todas_bases)
    
    for base in todas_bases:
        if tentativas >= max_tentativas:
            break
        tentativas += 1
        yield list(base)

def ler_arquivo_simplex(nome_arquivo):
    """Lê e interpreta o arquivo de entrada do problema de programação linear"""
    try:
        with open(nome_arquivo, "r", encoding="utf-8") as f:
            linhas = [linha.strip() for linha in f if linha.strip()]
    except FileNotFoundError:
        raise ValueError(f"Arquivo '{nome_arquivo}' não encontrado")

    if not linhas:
        raise ValueError("Arquivo vazio")

    # Processa função objetivo
    func_obj = linhas[0].replace(" ", "").lower()
    
    if func_obj.startswith("maxz="):
        eh_maximizacao = True
        expressao_obj = func_obj[5:]
    elif func_obj.startswith("minz="):
        eh_maximizacao = False
        expressao_obj = func_obj[5:]
    else:
        raise ValueError("Função objetivo deve começar com 'MaxZ=' ou 'MinZ='")

    # Extrai coeficientes da função objetivo
    termos = re.findall(r"([+-]?[\d.,]*)x(\d+)", expressao_obj)
    if not termos:
        raise ValueError("Nenhuma variável encontrada na função objetivo")

    coef_obj = {}
    for coef, var in termos:
        idx_var = int(var)
        # Converte coeficiente para float
        if not coef or coef == "+":
            val = 1.0
        elif coef == "-":
            val = -1.0
        else:
            val = float(coef.replace(",", "."))
        coef_obj[idx_var] = val

    num_var = max(coef_obj.keys())
    c = [coef_obj.get(i, 0.0) for i in range(1, num_var+1)]

    # Processa restrições
    restricoes = []
    b = []
    tipos_restricao = []

    for linha in linhas[1:]:  
        # Determina tipo de restrição
        if "<=" in linha:
            partes = linha.split("<=")
            tipo = "<="
        elif ">=" in linha:
            partes = linha.split(">=")
            tipo = ">="
        elif "=" in linha:
            partes = linha.split("=")
            tipo = "="
        else:
            raise ValueError(f"Restrição sem operador válido: '{linha}'")

        lhs, rhs = [p.strip() for p in partes]
        lhs = lhs.replace(" ", "")

        # Processa lado esquerdo da restrição
        coef_restr = [0.0]*num_var
        termos = re.findall(r"([+-]?[\d.,]*)x(\d+)", lhs)
        for coef, var in termos:
            idx = int(var) - 1
            if not coef or coef == "+":
                val = 1.0
            elif coef == "-":
                val = -1.0
            else:
                val = float(coef.replace(",", "."))
            if idx >= num_var:
                raise ValueError(f"Variável x{var} não definida na função objetivo")
            coef_restr[idx] = val

        # Processa lado direito
        try:
            valor_b = float(rhs.replace(",", "."))
        except ValueError:
            raise ValueError(f"Valor do lado direito inválido: '{rhs}'")

        restricoes.append(coef_restr)
        b.append(valor_b)
        tipos_restricao.append(tipo)

    # Adiciona variáveis de folga, excesso e artificiais
    num_colunas = num_var
    for i, tipo in enumerate(tipos_restricao):
        if tipo == "<=":
            for linha in restricoes:
                linha.append(0.0)
            restricoes[i][num_colunas] = 1.0
            c.append(0.0)
            num_colunas += 1
        elif tipo == ">=":
            for linha in restricoes:
                linha.extend([0.0, 0.0])
            restricoes[i][num_colunas] = -1.0
            restricoes[i][num_colunas+1] = 1.0
            c.extend([0.0, 1.0])
            num_colunas += 2
        elif tipo == "=":
            for linha in restricoes:
                linha.append(0.0)
            restricoes[i][num_colunas] = 1.0
            c.append(1.0)
            num_colunas += 1

    return eh_maximizacao, c, restricoes, b

def formatar_valor(v):
    """Formata valores numéricos para exibição"""
    return f"{v:.4f}".rstrip('0').rstrip('.') if '.' in f"{v:.4f}" else f"{v:.0f}"

def imprimir_matriz(matriz, nome="Matriz"):
    """Imprime matriz formatada"""
    print(f"\n{nome}:")
    if not matriz:
        print("[]")
        return
    
    for linha in matriz:
        print("[ " + " ".join(formatar_valor(elem).rjust(8) for elem in linha) + " ]")

def calcular_inversa(matriz):
    """Calcula matriz inversa usando eliminação de Gauss-Jordan"""
    n = len(matriz)
    if n == 0:
        raise ValueError("Matriz vazia")
    
    # Verifica se é quadrada
    for linha in matriz:
        if len(linha) != n:
            raise ValueError("Matriz deve ser quadrada para inversão")
    
    # Cria matriz aumentada [A|I]
    aumentada = [linha + [1.0 if j == i else 0.0 for j in range(n)] 
                for i, linha in enumerate(matriz)]
    
    for col in range(n):
        # Pivoteamento parcial
        max_row = max(range(col, n), key=lambda r: abs(aumentada[r][col]))
        if max_row != col:
            aumentada[col], aumentada[max_row] = aumentada[max_row], aumentada[col]
        
        pivot = aumentada[col][col]
        if abs(pivot) < 1e-10:
            raise ValueError("Matriz singular - não invertível")
        
        # Normaliza linha do pivô
        for j in range(col, 2*n):
            aumentada[col][j] /= pivot
        
        # Eliminação para outras linhas
        for i in range(n):
            if i != col and abs(aumentada[i][col]) > 1e-10:
                fator = aumentada[i][col]
                for j in range(col, 2*n):
                    aumentada[i][j] -= fator * aumentada[col][j]
    
    # Extrai a inversa
    return [linha[n:] for linha in aumentada]

def multiplicar_matrizes(A, B):
    """Multiplica duas matrizes A (m×n) e B (n×p)"""
    if not A or not B:
        raise ValueError("Matrizes vazias")
    
    linhas_A, cols_A = len(A), len(A[0])
    linhas_B, cols_B = len(B), len(B[0])
    
    if cols_A != linhas_B:
        raise ValueError("Dimensões incompatíveis para multiplicação")
    
    return [[sum(A[i][k] * B[k][j] for k in range(cols_A)) 
            for j in range(cols_B)] for i in range(linhas_A)]

def resolver_sistema(B, b, max_tentativas=3):
    """Resolve o sistema Bx = b com tratamento de erros numéricos"""
    for tentativa in range(max_tentativas):
        try:
            B_inv = calcular_inversa(B)
            x = multiplicar_matrizes(B_inv, [[bi] for bi in b])
            return [row[0] for row in x]
        except Exception as e:
            if tentativa == max_tentativas - 1:
                raise ValueError(f"Falha ao resolver sistema após {max_tentativas} tentativas: {str(e)}")
            # Adiciona pequena perturbação para tentar resolver problemas numéricos
            B = [[elem + random.uniform(-1e-8, 1e-8) for elem in linha] for linha in B]

def fase_I_aleatoria(A, b, c, max_bases=100):
    """Implementa a Fase I do Simplex com seleção aleatória de bases"""
    m, n = len(A), len(A[0])
    
    # Adiciona variáveis artificiais
    A_art = [linha.copy() + [1.0 if i == j else 0.0 for j in range(m)] 
             for i, linha in enumerate(A)]
    c_art = c.copy() + [1.0]*m  # Penaliza variáveis artificiais
    
    print("\n=== FASE I ===")
    print(f"Procurando base factível entre {min(max_bases, len(list(combinations(range(n+m), m))))} possibilidades...")
    
    for base_candidata in gerenciar_bases_aleatorias(n + m, m, max_bases):
        try:
            B = [[A_art[i][j] for j in base_candidata] for i in range(m)]
            if np.linalg.matrix_rank(B) < m:
                continue  # Ignora bases linearmente dependentes
                
            x_B = resolver_sistema(B, b)
            
            # Verifica factibilidade
            if all(x >= -1e-6 for x in x_B):
                # Remove variáveis artificiais da base se possível
                base_final = []
                art_indices = set(range(n, n+m))
                
                for var in base_candidata:
                    if var < n:  # Variável original
                        base_final.append(var)
                    else:        # Variável artificial
                        # Tenta substituir por variável original
                        substituida = False
                        for j in range(n):
                            if j not in base_candidata:
                                try:
                                    y = resolver_sistema(B, [A_art[i][j] for i in range(m)])
                                    if abs(y[base_candidata.index(var)]) > 1e-6:
                                        base_final.append(j)
                                        substituida = True
                                        break
                                except:
                                    continue
                        if not substituida:
                            base_final.append(var)
                
                if len(base_final) == m:
                    print(f"Base factível encontrada: {[x+1 for x in base_final]}")
                    return base_final
        except:
            continue
    
    raise ValueError("Não foi encontrada base factível na Fase I")

def simplex_core(A, b, c, base, nao_base, eh_fase_I=False):
    """Implementa o núcleo do algoritmo Simplex"""
    m = len(A)
    
    while True:
        try:
            # Passo 1: Resolve sistema Bx_B = b
            B = [[A[i][j] for j in base] for i in range(m)]
            x_B = resolver_sistema(B, b)
            
            # Passo 2: Calcula vetor de multiplicadores simplex
            c_B = [c[j] for j in base]
            B_inv = calcular_inversa(B)
            lambda_T = [sum(c_B[k] * B_inv[k][i] for k in range(m)) for i in range(m)]
            
            # Passo 3: Calcula custos reduzidos
            c_hat = []
            for j in nao_base:
                a_j = [A[i][j] for i in range(m)]
                c_hat.append(c[j] - sum(lambda_T[i] * a_j[i] for i in range(m)))
            
            # Verifica otimalidade
            if all(ch >= -1e-6 for ch in c_hat):
                valor = sum(c_B[i] * x_B[i] for i in range(m))
                return {'status': 'otimo', 'base': base, 'x_B': x_B, 'valor': valor}
            
            # Seleciona variável para entrar na base
            k = next((i for i, ch in enumerate(c_hat) if ch < -1e-6), None)
            if k is None:
                return {'status': 'erro', 'mensagem': 'Nenhuma variável candidata'}
            
            # Calcula direção simplex
            a_Nk = [A[i][nao_base[k]] for i in range(m)]
            y = resolver_sistema(B, a_Nk)
            
            # Determina variável para sair da base
            epsilon, l = float('inf'), -1
            for i in range(m):
                if y[i] > 1e-6:
                    ratio = x_B[i] / y[i]
                    if ratio < epsilon - 1e-6:
                        epsilon, l = ratio, i
            
            if l == -1:
                return {'status': 'ilimitado', 'base': base, 'x_B': x_B}
            
            # Atualiza base
            entrada = nao_base[k]
            saida = base[l]
            base[l] = entrada
            nao_base[k] = saida
            
        except Exception as e:
            return {'status': 'erro', 'mensagem': str(e)}

def executar_simplex(eh_maximizacao, c, A, b):
    """Executa o método Simplex de duas fases"""
    print("\n=== PROBLEMA ORIGINAL ===")
    print("Tipo:", "Maximização" if eh_maximizacao else "Minimização")
    imprimir_matriz([c], "Vetor de custos (c)")
    imprimir_matriz(A, "Matriz de restrições (A)")
    imprimir_matriz([[bi] for bi in b], "Vetor de recursos (b)")
    
    # Fase I - Encontra base factível
    try:
        base = fase_I_aleatoria(A, b, c)
        nao_base = [j for j in range(len(A[0])) if j not in base]
        print("\nFASE I concluída. Base factível inicial:", [var+1 for var in base])
    except ValueError as e:
        print(f"\nErro na Fase I: {str(e)}")
        return
    
    # Fase II - Resolve problema original
    print("\n=== FASE II ===")
    c_original = [-x if eh_maximizacao else x for x in c]  # Padroniza como minimização
    resultado = simplex_core(A, b, c_original, base, nao_base)
    
    # Processa resultados
    if resultado['status'] == 'otimo':
        x = [0.0] * len(A[0])
        for i, var in enumerate(resultado['base']):
            x[var] = resultado['x_B'][i]
        
        valor_otimo = sum(c[j] * x[j] for j in range(len(c)))
        if eh_maximizacao:
            valor_otimo *= -1
        
        print("\n=== SOLUÇÃO ÓTIMA ===")
        print(f"Valor ótimo: {formatar_valor(valor_otimo)}")
        print("\nVariáveis:")
        for j, val in enumerate(x):
            print(f"x{j+1} = {formatar_valor(val)}")
        
    elif resultado['status'] == 'ilimitado':
        print("\nO problema é ilimitado")
    else:
        print("\nErro:", resultado.get('mensagem', ''))

if __name__ == "__main__":
    try:
        eh_max, c, A, b = ler_arquivo_simplex("teste.txt")
        executar_simplex(eh_max, c, A, b)
    except Exception as e:
        print(f"\nErro: {str(e)}")