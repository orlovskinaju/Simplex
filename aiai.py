import re
import random
import numpy as np
from itertools import combinations

def formatar_valor(v):
    return f"{v:.4f}".rstrip('0').rstrip('.') if '.' in f"{v:.4f}" else f"{v:.0f}"

def imprimir_matriz(matriz, nome="Matriz"):
    print(f"\n{nome}:")
    if not matriz:
        print("[]")
        return
    for linha in matriz:
        print("[ " + " ".join(formatar_valor(elem).rjust(8) for elem in linha) + " ]")

def gerenciar_bases_aleatorias(num_colunas, num_restricoes, max_tentativas=None):
    if num_colunas < num_restricoes:
        raise ValueError("Número de colunas deve ser ≥ número de restrições")

    todas_bases = list(combinations(range(num_colunas), num_restricoes))
    random.shuffle(todas_bases)

    tentativas = 0
    max_tentativas = max_tentativas or len(todas_bases)

    for base in todas_bases:
        if tentativas >= max_tentativas:
            break
        tentativas += 1
        yield list(base)

def ler_arquivo_simplex(nome_arquivo):
    """Lê o problema de programação linear de um arquivo"""
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
        if "<=" in linha:
            partes = linha.split("<="), "<="
        elif ">=" in linha:
            partes = linha.split(">="), ">="
        elif "=" in linha:
            partes = linha.split("="), "="
        else:
            raise ValueError(f"Restrição sem operador válido: '{linha}'")

        (lhs, rhs), tipo = partes
        lhs = lhs.replace(" ", "")

        # Extrai coeficientes da restrição
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

        try:
            valor_b = float(rhs.replace(",", "."))
        except ValueError:
            raise ValueError(f"Valor do lado direito inválido: '{rhs}'")

        restricoes.append(coef_restr)
        b.append(valor_b)
        tipos_restricao.append(tipo)

    # Transforma restrições de igualdade em duas desigualdades
    novas_restricoes = []
    novos_tipos = []
    novo_b = []
    
    for i, tipo in enumerate(tipos_restricao):
        if tipo == "=":
            # Divide em duas restrições (<= e >=)
            novas_restricoes.append(restricoes[i].copy())
            novos_tipos.append("<=")
            novo_b.append(b[i])
            
            novas_restricoes.append(restricoes[i].copy())
            novos_tipos.append(">=")
            novo_b.append(b[i])
        else:
            novas_restricoes.append(restricoes[i].copy())
            novos_tipos.append(tipo)
            novo_b.append(b[i])
    
    restricoes = novas_restricoes
    tipos_restricao = novos_tipos
    b = novo_b

    # Adiciona variáveis de folga/excesso/artificiais
    num_colunas = num_var
    for i, tipo in enumerate(tipos_restricao):
        if tipo == "<=":
            for linha in restricoes:
                linha.append(0.0)
            restricoes[i][num_colunas] = 1.0  # Variável de folga
            c.append(0.0)
            num_colunas += 1
        elif tipo == ">=":
            for linha in restricoes:
                linha.extend([0.0, 0.0])
            restricoes[i][num_colunas] = -1.0  # Variável de excesso
            restricoes[i][num_colunas+1] = 1.0  # Variável artificial
            c.extend([0.0, 0.0])
            num_colunas += 2

    return eh_maximizacao, c, restricoes, b

def calcular_inversa(matriz):
    n = len(matriz)
    if n == 0:
        raise ValueError("Matriz vazia")
    for linha in matriz:
        if len(linha) != n:
            raise ValueError("Matriz deve ser quadrada para inversão")
    
    aumentada = [linha + [1.0 if j == i else 0.0 for j in range(n)] 
                for i, linha in enumerate(matriz)]
    
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aumentada[r][col]))
        if max_row != col:
            aumentada[col], aumentada[max_row] = aumentada[max_row], aumentada[col]
        
        pivot = aumentada[col][col]
        if abs(pivot) < 1e-10:
            raise ValueError("Matriz singular - não invertível")
        
        for j in range(col, 2*n):
            aumentada[col][j] /= pivot
            
        for i in range(n):
            if i != col and abs(aumentada[i][col]) > 1e-10:
                fator = aumentada[i][col]
                for j in range(col, 2*n):
                    aumentada[i][j] -= fator * aumentada[col][j]
    
    return [linha[n:] for linha in aumentada]

def multiplicar_matrizes(A, B):
    if not A or not B:
        raise ValueError("Matrizes vazias")
    
    linhas_A, cols_A = len(A), len(A[0])
    linhas_B, cols_B = len(B), len(B[0])
    
    if cols_A != linhas_B:
        raise ValueError("Dimensões incompatíveis para multiplicação")
    
    return [[sum(A[i][k] * B[k][j] for k in range(cols_A)) 
            for j in range(cols_B)] for i in range(linhas_A)]

def resolver_sistema(B, b, max_tentativas=10):
    for tentativa in range(max_tentativas):
        try:
            B_inv = calcular_inversa(B)
            x = multiplicar_matrizes(B_inv, [[bi] for bi in b])
            return [row[0] for row in x]
        except Exception as e:
            if tentativa == max_tentativas - 1:
                raise ValueError(f"Falha ao resolver sistema após {max_tentativas} tentativas: {str(e)}")
            B = [[elem + random.uniform(-1e-5, 1e-5) for elem in linha] for linha in B]
def fase_I_aleatoria(A, b, c, max_bases=100):
    m, n = len(A), len(A[0])
    
    # Passo 1: Identificar restrições que precisam de artificiais
    precisa_artificial = []
    for i in range(m):
        # Restrições >= ou = precisam de artificiais
        if any(A[i][j] < -1e-6 for j in range(n)) or sum(abs(A[i][j]) for j in range(n)) == 0:
            precisa_artificial.append(True)
        else:
            precisa_artificial.append(False)
    
    # Passo 2: Adicionar variáveis artificiais somente onde necessário
    A_art = [linha.copy() for linha in A]
    num_artificiais = sum(precisa_artificial)
    indices_artificiais = []
    
    col_atual = n
    for i in range(m):
        if precisa_artificial[i]:
            for linha in A_art:
                linha.append(1.0 if len(linha) == col_atual else 0.0)
            indices_artificiais.append(col_atual)
            col_atual += 1
        else:
            for linha in A_art:
                linha.append(0.0)  # Espaço reservado, mas não é artificial
    
    c_art = [0.0]*n + [1.0]*num_artificiais
    
    print("\n=== FASE I ===")
    print(f"Variáveis artificiais adicionadas nas restrições: {[i+1 for i, val in enumerate(precisa_artificial) if val]}")
    
    # Passo 3: Busca por base factível com verificação rigorosa
    for base_candidata in gerenciar_bases_aleatorias(n + num_artificiais, m, max_bases):
        try:
            B = [[A_art[i][j] for j in base_candidata] for i in range(m)]
            if np.linalg.matrix_rank(B) < m:
                continue
                
            x_B = resolver_sistema(B, b)
            
            # Verificação EXTRA-RIGOROSA do Caso A
            todas_artificiais_zero = True
            for i, var in enumerate(base_candidata):
                if var in indices_artificiais:
                    if x_B[i] > 1e-6:  # Qualquer artificial > 0 → inviável
                        todas_artificiais_zero = False
                        break
            
            if not todas_artificiais_zero:
                continue  # Descarta base - Caso B potencial
                
            # Verifica factibilidade nas restrições ORIGINAIS
            solucao = [0.0] * n
            for i, var in enumerate(base_candidata):
                if var < n:  # Ignora artificiais
                    solucao[var] = x_B[i]
            
            factivel = True
            for i in range(m):
                lhs = sum(A[i][j] * solucao[j] for j in range(n))
                if precisa_artificial[i]:  # Restrição original era >= ou =
                    if lhs < b[i] - 1e-6:  # Não satisfaz
                        factivel = False
                        break
                else:  # Restrição original era <=
                    if lhs > b[i] + 1e-6:  # Não satisfaz
                        factivel = False
                        break
            
            if factivel:  # CASO A CONFIRMADO
                print(f"Base factível encontrada: {[x+1 for x in base_candidata if x < n]}")
                return [var for var in base_candidata if var < n]  # Remove artificiais
            
        except:
            continue
    
    # CASO B - Nenhuma base factível encontrada
    print("\n=== RESULTADO FASE I ===")
    print("Todas as bases testadas resultaram em:")
    print("1. Variáveis artificiais positivas OU")
    print("2. Não satisfazem restrições originais")
    raise ValueError("PROBLEMA INFACTÍVEL (Caso B) - Não existe solução viável")

def simplex(A, b, c, base, nao_base, eh_fase_I=False):
    m = len(A)
    iteracoes = 0
    max_iteracoes = 1000  # Aumentado para problemas mais complexos

    while iteracoes < max_iteracoes:
        iteracoes += 1
        try:
            B = [[A[i][j] for j in base] for i in range(m)]
            x_B = resolver_sistema(B, b)
            c_B = [c[j] for j in base]
            B_inv = calcular_inversa(B)
            lambda_T = [sum(c_B[k] * B_inv[k][i] for k in range(m)) for i in range(m)]
            c_hat = []
            for j in nao_base:
                a_j = [A[i][j] for i in range(m)]
                c_hat.append(c[j] - sum(lambda_T[i] * a_j[i] for i in range(m)))
            
            if all(ch >= -1e-6 for ch in c_hat):
                valor = sum(c_B[i] * x_B[i] for i in range(m))
                return {'status': 'otimo', 'base': base, 'x_B': x_B, 'valor': valor}
            
            k = next((i for i, ch in enumerate(c_hat) if ch < -1e-6), None)
            if k is None:
                return {'status': 'erro', 'mensagem': 'Nenhuma variável candidata'}
            
            a_Nk = [A[i][nao_base[k]] for i in range(m)]
            y = resolver_sistema(B, a_Nk)
            
            # Verificação robusta de ilimitação
            if all(y_i <= 1e-6 for y_i in y):
                # Verifica se a variável pode crescer indefinidamente
                solucao = [0.0] * len(A[0])
                for i, var in enumerate(base):
                    solucao[var] = x_B[i]
                
                # Testa se a direção leva a valores factíveis para todas as restrições
                ilimitado = True
                for i in range(m):
                    if sum(A[i][j] * (solucao[j] + (1 if j == nao_base[k] else 0)) for j in range(len(A[0]))) > b[i] + 1e-6:
                        ilimitado = False
                        break
                
                if ilimitado:
                    print("\n=== PROBLEMA ILIMITADO ===")
                    print(f"Variável x{nao_base[k]+1} pode crescer indefinidamente")
                    return {'status': 'ilimitado', 'direcao': nao_base[k]}
            
            epsilon, l = float('inf'), -1
            for i in range(m):
                if y[i] > 1e-6:
                    ratio = x_B[i] / y[i]
                    if ratio < epsilon - 1e-6:
                        epsilon, l = ratio, i
            
            if l == -1:
                return {'status': 'ilimitado', 'base': base, 'x_B': x_B}
            
            entrada = nao_base[k]
            saida = base[l]
            base[l] = entrada
            nao_base[k] = saida
            
        except Exception as e:
            print(f"Erro na iteração {iteracoes}: {str(e)} - Tentando nova base...")
            todas_colunas = list(range(len(A[0])))
            random.shuffle(todas_colunas)
            base = todas_colunas[:m]
            nao_base = [j for j in todas_colunas if j not in base]
            continue

    return {'status': 'erro', 'mensagem': 'Número máximo de iterações atingido'}
    
def executar_simplex(eh_maximizacao, c, A, b):
    print("\n=== PROBLEMA ORIGINAL ===")
    print("Tipo:", "Maximização" if eh_maximizacao else "Minimização")
    imprimir_matriz([c], "Vetor de custos (c)")
    imprimir_matriz(A, "Matriz de restrições (A)")
    imprimir_matriz([[bi] for bi in b], "Vetor de recursos (b)")

    c_original = c.copy()
    num_var_originais = len(c_original)

    try:
        # === FASE I ===
        print("\n=== FASE I - BUSCA POR SOLUÇÃO FACTÍVEL ===")
        base = fase_I_aleatoria(A, b, c)
        nao_base = [j for j in range(len(A[0])) if j not in base]
        print("\nFASE I concluída. Base factível inicial:", [var+1 for var in base])

        # Remove variáveis artificiais para Fase II
        A = [linha[:num_var_originais] for linha in A]
        c = c_original.copy()
        base = [j for j in base if j < num_var_originais]
        nao_base = [j for j in nao_base if j < num_var_originais]

        # === FASE II ===
        print("\n=== FASE II - OTIMIZAÇÃO ===")
        
        resultado = simplex(A, b, c, base, nao_base, eh_maximizacao)

        if resultado['status'] == 'otimo':
            x = [0.0] * num_var_originais
            for i, var in enumerate(resultado['base']):
                if var < num_var_originais:
                    x[var] = resultado['x_B'][i]
            
            valor_otimo = sum(c_original[j] * x[j] for j in range(num_var_originais))

            print("\n=== SOLUÇÃO ÓTIMA ENCONTRADA ===")
            print(f"Valor da função objetivo: {formatar_valor(valor_otimo)}")
            print("\nVariáveis básicas:")
            for j, val in enumerate(x):
                print(f"x{j+1} = {formatar_valor(val)}")

        elif resultado['status'] == 'ilimitado':
            print("\nO problema é ilimitado na direção de:")
            print(f"Variável x{resultado['direcao']+1} pode crescer indefinidamente")
            if eh_maximizacao:
                print("(Valor objetivo tende a +∞)")
            else:
                print("(Valor objetivo tende a -∞)")

    except ValueError as e:
        print(f"\nERRO: {str(e)}")

if __name__ == "__main__":
    try:
        eh_max, c, A, b = ler_arquivo_simplex("teste.txt")
        executar_simplex(eh_max, c, A, b)
    except Exception as e:
        print(f"\nErro: {str(e)}")