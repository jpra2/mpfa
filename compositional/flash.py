import numpy as np
import math

def flash_rr(z, K, max_iter=1000, tol=1e-8):
    """
        Realiza o cálculo do flash para duas fases (óleo e gás) e uma quantidade genérica de componentes
        utilizando o método de Rachford-Rice.
        
        Parâmetros:
        z : array
            Fração molar de cada componente na alimentação.
        K : array
            Coeficiente de distribuição de cada componente entre as fases.
        max_iter : int, opcional
            Número máximo de iterações para o método de Rachford-Rice.
        tol : float, opcional
            Tolerância para o critério de convergência do método de Rachford-Rice.
            
        Retorno:
        x : array
            Fração molar de cada componente na fase líquida.
        y : array
            Fração molar de cada componente na fase gasosa.
    """
    
    # Verifica se a soma dos coeficientes de distribuição é igual ao número de componentes
    if abs(np.sum(K) - len(K)) > 1e-8:
        raise ValueError('A soma dos coeficientes de distribuição deve ser igual ao número de componentes.')
    
    # Define a função do método de Rachford-Rice
    def rr_func(beta):
        f = np.zeros(len(K))
        for i in range(len(K)):
            f[i] = z[i] * (1 - beta) / (1 + beta * (K[i] - 1))
        return np.sum(f)
    
    # Define a derivada da função do método de Rachford-Rice
    def rr_deriv(beta):
        df = np.zeros(len(K))
        for i in range(len(K)):
            df[i] = -z[i] * (1 - K[i]) / (1 + beta * (K[i] - 1)) ** 2
        return np.sum(df)
    
    # Aplica o método de Newton-Raphson para encontrar a raiz da função do método de Rachford-Rice
    beta = 0.5
    for i in range(max_iter):
        f = rr_func(beta)
        df = rr_deriv(beta)
        beta -= f / df
        if abs(f) < tol:
            break
    
    # Calcula as frações molares das fases líquida e gasosa
    x = np.zeros(len(K))
    y = np.zeros(len(K))
    for i in range(len(K)):
        x[i] = z[i] * (1 - beta) / (1 + beta * (K[i] - 1))
        y[i] = K[i] * x[i]
    
    # Retorna as frações molares das fases líquida e gasosa
    return x, y

def initial_K(z, nc):
    """Estimativa inicial do valor de K

    Args:
        z (array): Frações molares dos componentes
        nc (int): Número de componentes

    Returns:
        array: Estimativa de K
    """
    K = np.zeros(nc)
    for i in range(nc):
        if z[i] < 0.5:
            K[i] = 1e-10
        else:
            K[i] = 1e10
    beta = 0
    for i in range(nc):
        beta += z[i] * (K[i] - 1) / (1 + beta * (K[i] - 1))
    while abs(beta) > 1e-8:
        f_K = lambda K: sum(z * (K - 1) / (1 + beta * (K - 1))) 
        dKdbeta = lambda K: -sum(z * (K - 1)**2 / (1 + beta * (K - 1))**2)
        K_new = K - f_K(K) / dKdbeta(K)
        beta_new = 0
        for i in range(nc):
            beta_new += z[i] * (K_new[i] - 1) / (1 + beta * (K_new[i] - 1))
        K = K_new
        beta = beta_new
    return K



def peng_robinson_K(T, P, Pc, Tc, w):
    # Cálculo dos parâmetros da equação de estado
    Tr = T / Tc
    k = 0.37464 + 1.54226 * w - 0.26992 * w ** 2
    alpha = (1 + k * (1 - math.sqrt(Tr))) ** 2
    a = 0.45724 * (R * Tc) ** 2 / Pc * alpha
    b = 0.07780 * R * Tc / Pc

    # Cálculo do fator de compressibilidade
    A = a * P / (R * T) ** 2
    B = b * P / (R * T)
    coeficientes = [1, -(1 - B), A - 2 * B - 3 * B ** 2, -(A * B - B ** 2 - B ** 3)]
    Z = np.roots(coeficientes)
    Z = Z.real[Z.imag == 0][0]  # Seleciona a raiz real positiva

    # Cálculo dos parâmetros do componente
    Vm = (R * T) / (P * (Z - B))
    Bi = b * Vm / Vc
    factor = (2 / math.sqrt(alpha)) - (2 / (math.sqrt(alpha) + b))

    # Cálculo do coeficiente de distribuição
    ln_K = (Bi - b) / Vm - math.log(Z / (Z - 1)) + (A / (math.sqrt(8) * Bi)) * factor
    K = math.exp(ln_K)
    
    return K

