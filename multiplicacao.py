def multiplicar_matrizes(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Número de colunas de A deve ser igual ao número de linhas de B.")

    resultado = []
    i = 0
    while i < len(A):
        linha = []
        j = 0
        while j < len(B[0]):
            soma = 0
            k = 0
            while k < len(B):
                soma += A[i][k] * B[k][j]
                k += 1
            linha.append(soma)
            j += 1
        resultado.append(linha)
        i += 1

    return resultado

resultado = multiplicar_matrizes()

i = 0
while i < len(resultado):
    print(resultado[i])
    i += 1
