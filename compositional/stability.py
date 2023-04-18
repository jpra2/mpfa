import numpy as np

# Definição das constantes e parâmetros do modelo de Peng-Robinson
R = 8.314 # J/mol.K
Tc = np.array([369.83, 425.12, 617.7, 647.14]) # K
Pc = np.array([48.72, 33.67, 22.09, 22.06]) * 1e5 # Pa
w = np.array([0.152, 0.199, 0.49, 0.344]) # Fator acêntrico
kij = np.array([[0.0, -0.01, 0.0, 0.0], [-0.01, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]) # Parâmetros de interação

# Cálculo dos parâmetros do modelo de Peng-Robinson
Tr = T / Tc
alpha = (1 + (0.37464 + 1.54226 * w - 0.26992 * w ** 2) * (1 - np.sqrt(Tr))) ** 2
a = 0.45724 * alpha * (R * Tc) ** 2 / Pc
b = 0.07780 * R * Tc / Pc

# Cálculo dos coeficientes de fugacidade utilizando o modelo de Peng-Robinson
def fugacity_coefficient(P, T, x, a, b, kij):
    phi = np.zeros_like(x)
    ln_phi = np.zeros_like(x)
    A = a * P / (R * T) ** 2
    B = b * P / (R * T)
    Am = np.sum(np.sqrt(A * x))
    bm = np.sum(B * x)
    for i in range(len(x)):
        ln_phi[i] = ((B[i] / bm) * (Z - 1) - np.log(Z - B)) + (Am / bm) * (B[i] / bm - 2 * np.sum(A * np.sqrt(x)) / Am) * np.log((Z + bm) / Z)
        phi[i] = np.exp(ln_phi[i])
    return phi

# Cálculo da energia livre de Gibbs da mistura
def gibbs_energy(P, T, x, a, b, kij):
    Z = np.roots([1, -1, Am - B - B ** 2, -Am * B])[np.argmin(np.abs(np.imag(np.roots([1, -1, Am - B - B ** 2, -Am * B]))))]
    ln_phi = np.log(fugacity_coefficient(P, T, x, a, b, kij))
    G_mix = R * T * np.sum(x * ln_phi)
    return G_mix

# Cálculo do ponto crítico da mistura
def critical_point(Tc, Pc, w):
    a = 0.45724 * (R * Tc) ** 2 / Pc
    b = 0.07780 * R * Tc / Pc
    kappa = 0.37464 + 1.54226 * w - 0.26992 * w ** 2
                                      

# import numpy as np

# Função que calcula a energia livre de Gibbs da mistura
def gibbs_energy(P, T, x, a, b, kij):
    # Cálculo dos parâmetros de interação
    A = np.sqrt(a[:, None] * a) * (1 - kij)
    B = b[:, None] * b
    # Cálculo das frações molares iniciais
    y = x
    # Definição de um erro máximo e de um erro inicial maior que o erro máximo
    eps = 1e-6
    error = 1
    # Loop de cálculo das frações molares das fases
    while error > eps:
        # Cálculo das matrizes de interação
        G = np.exp(-A / (R * T))
        # Cálculo da matriz K
        K = G * np.outer(y, y) / np.dot(y, G * y)
        # Cálculo das novas frações molares
        y_new = K.dot(x)
        # Cálculo do erro
        error = np.max(np.abs(y_new - y))
        # Atualização das frações molares
        y = y_new
    # Cálculo da energia livre de Gibbs das fases
    ln_phi = np.log(y) + np.outer(np.sum(K, axis=1), (1 - np.sum(y, axis=0)))
    phi = np.exp(ln_phi)
    G_mix = R * T * np.sum(x * ln_phi)
    G_phase = R * T * np.sum(y * ln_phi, axis=0)
    # Cálculo dos índices de seletividade
    S = np.log(K) + np.outer(np.sum(K, axis=1), (1 - np.sum(y, axis=0)))
    # Cálculo da matriz de coordenadas reduzidas
    r = np.log(y / phi) + S.dot(y - x)
    # Verificação das fases existentes
    if np.all(y > eps):
        phases = ['Single phase']
    elif np.all(y < 1 - eps):
        phases = ['Single phase']
    else:
        phases = ['Liquid phase', 'Vapor phase']
    # Cálculo das frações molares das fases
    x_phase = [x, x]
    y_phase = [y, y]
    for i in range(2):
        if phases[i] == 'Liquid phase':
            x_phase[i] = y_phase[i]
        elif phases[i] == 'Vapor phase':
            x_phase[i] = x
    # Retorno dos resultados
    return phases, x_phase, y_phase, G_mix, G_phase, r

def estabilidade_2():
    import numpy as np
    from pycame.equilibrium import flash

    # Definir as condições iniciais como arrays
    P = np.array([1.0, 2.0, 3.0])  # Pressão (bar)
    T = np.array([300.0, 400.0, 500.0])  # Temperatura (K)
    z = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])  # Frações molares iniciais

    # Calcular a estabilidade de fases para todas as condições iniciais
    results = flash(z, T, P)

    # Iterar sobre os resultados para cada condição inicial
    for i, result in enumerate(results):
        print(f'Condição inicial {i+1}:')
        # Verificar as fases estáveis
        print('Fases estáveis:', result.phase_fractions.keys())

        # Imprimir as frações molares de cada componente em cada fase estável
        for phase, frac in result.phase_fractions.items():
            print(f'Frações molares da fase {phase}:')
            for comp, comp_frac in frac.items():
                print(f'{comp}: {comp_frac:.6f}')
