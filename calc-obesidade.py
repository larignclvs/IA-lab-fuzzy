import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# Variáveis de entrada (quantidade de comida em kcal)
comida = ctrl.Antecedent(np.arange(0, 11, 1), 'comida')

# Variáveis de saída (peso ganho em kg)
peso = ctrl.Consequent(np.arange(0, 11, 1), 'peso')

# ----- TRIANGULAR -----
comida['pouco_tri'] = fuzz.trimf(comida.universe, [0, 2, 4])
comida['razoavel_tri'] = fuzz.trimf(comida.universe, [2, 5, 8])
comida['bastante_tri'] = fuzz.trimf(comida.universe, [6, 8, 10])

peso['leve_tri'] = fuzz.trimf(peso.universe, [0, 2, 4])
peso['medio_tri'] = fuzz.trimf(peso.universe, [2, 5, 8])
peso['pesado_tri'] = fuzz.trimf(peso.universe, [6, 8, 10])

# ----- GAUSSIANA -----
comida['pouco_gauss'] = fuzz.gaussmf(comida.universe, 2, 1)
comida['razoavel_gauss'] = fuzz.gaussmf(comida.universe, 5, 1.5)
comida['bastante_gauss'] = fuzz.gaussmf(comida.universe, 8, 1)

peso['leve_gauss'] = fuzz.gaussmf(peso.universe, 2, 1)
peso['medio_gauss'] = fuzz.gaussmf(peso.universe, 5, 1.5)
peso['pesado_gauss'] = fuzz.gaussmf(peso.universe, 8, 1)

# ----- TRAPEZOIDAL -----
comida['pouco_trap'] = fuzz.trapmf(comida.universe, [0, 1, 3, 4])
comida['razoavel_trap'] = fuzz.trapmf(comida.universe, [2, 4, 6, 8])
comida['bastante_trap'] = fuzz.trapmf(comida.universe, [6, 7, 9, 10])

peso['leve_trap'] = fuzz.trapmf(peso.universe, [0, 1, 3, 4])
peso['medio_trap'] = fuzz.trapmf(peso.universe, [2, 4, 6, 8])
peso['pesado_trap'] = fuzz.trapmf(peso.universe, [6, 7, 9, 10])

# Criando conjuntos de regras para cada tipo de função
def criar_sistema_regras(tipo):
    if tipo == 'triangular':
        regras = [
            ctrl.Rule(comida['pouco_tri'], peso['leve_tri']),
            ctrl.Rule(comida['razoavel_tri'], peso['medio_tri']),
            ctrl.Rule(comida['bastante_tri'], peso['pesado_tri'])
        ]
    elif tipo == 'gaussiana':
        regras = [
            ctrl.Rule(comida['pouco_gauss'], peso['leve_gauss']),
            ctrl.Rule(comida['razoavel_gauss'], peso['medio_gauss']),
            ctrl.Rule(comida['bastante_gauss'], peso['pesado_gauss'])
        ]
    elif tipo == 'trapezoidal':
        regras = [
            ctrl.Rule(comida['pouco_trap'], peso['leve_trap']),
            ctrl.Rule(comida['razoavel_trap'], peso['medio_trap']),
            ctrl.Rule(comida['bastante_trap'], peso['pesado_trap'])
        ]
    else:
        raise ValueError("Tipo inválido. Escolha entre 'triangular', 'gaussiana' ou 'trapezoidal'.")

    sistema = ctrl.ControlSystem(regras)
    return ctrl.ControlSystemSimulation(sistema)

# Testando diferentes tipos de funções de pertinência
tipos = ['triangular', 'gaussiana', 'trapezoidal']
valores_teste = [2, 5, 9]

for tipo in tipos:
    print(f"\n--- Testando {tipo.upper()} ---")
    calculadora = criar_sistema_regras(tipo)
    for valor in valores_teste:
        calculadora.input['comida'] = valor
        calculadora.compute()
        print(f'Para {valor} kcal, o peso previsto é: {calculadora.output["peso"]:.2f} kg')

def plotar_funcoes(tipo):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle(f"Funções de Pertinência - {tipo.capitalize()}", fontsize=14)

    # Escolhendo as funções corretas para cada tipo
    if tipo == 'triangular':
        conjuntos_comida = ['pouco_tri', 'razoavel_tri', 'bastante_tri']
        conjuntos_peso = ['leve_tri', 'medio_tri', 'pesado_tri']
    elif tipo == 'gaussiana':
        conjuntos_comida = ['pouco_gauss', 'razoavel_gauss', 'bastante_gauss']
        conjuntos_peso = ['leve_gauss', 'medio_gauss', 'pesado_gauss']
    elif tipo == 'trapezoidal':
        conjuntos_comida = ['pouco_trap', 'razoavel_trap', 'bastante_trap']
        conjuntos_peso = ['leve_trap', 'medio_trap', 'pesado_trap']
    
    # Plotando a função de pertinência da entrada (comida)
    for conjunto in conjuntos_comida:
        ax[0].plot(comida.universe, comida[conjunto].mf, label=conjunto)
    ax[0].set_title("Comida (Entrada)")
    ax[0].legend()

    # Plotando a função de pertinência da saída (peso)
    for conjunto in conjuntos_peso:
        ax[1].plot(peso.universe, peso[conjunto].mf, label=conjunto)
    ax[1].set_title("Peso (Saída)")
    ax[1].legend()

    plt.show()

# Chamando a função para cada tipo de pertinência
for tipo in tipos:
    plotar_funcoes(tipo)


