import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime

# ========= Funções de Ativação e Derivadas =========
def leaky_relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, Z * alpha)

def leaky_relu_backward(dA, Z, alpha=0.01):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] *= alpha
    return dZ

def linear(Z):
    return Z

def linear_backward(dA, Z):
    return dA

# ========= Geração de Dados com One-Hot Encoding =========
def gerar_dados_curvas_one_hot(num_curvas, r1, r2, y_v_min, y_v_max):
    X_one_hot = np.identity(num_curvas)
    np.random.seed(42)
    y_vertices = np.random.uniform(y_v_min, y_v_max, num_curvas).reshape(1, -1)
    
    parabol_params_reais = []
    for i in range(num_curvas):
        y_v_atual = y_vertices[0, i]
        if (r1 - r2) == 0: 
            a_parabola, b_parabola, c_parabola = 0,0,y_v_atual 
        else:
            a_parabola = (-4 * y_v_atual) / ((r1 - r2)**2)
            b_parabola = -a_parabola * (r1 + r2)
            c_parabola = a_parabola * r1 * r2
        parabol_params_reais.append({'a': a_parabola, 'b': b_parabola, 'c': c_parabola, 'y_v': y_v_atual, 'idx': i})
        
    return X_one_hot.T, y_vertices, parabol_params_reais

# ========= Normalização =========
def normalizar_dados(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def desnormalizar_dados(normalized_data, mean, std):
    return normalized_data * std + mean

# ========= Estrutura da Rede =========
def inicializar_parametros(camadas_dims):
    np.random.seed(42)
    parametros = {}
    L = len(camadas_dims)
    for l in range(1, L):
        parametros[f'W{l}'] = np.random.randn(camadas_dims[l], camadas_dims[l-1]) * np.sqrt(2. / camadas_dims[l-1])
        parametros[f'b{l}'] = np.zeros((camadas_dims[l], 1))
    return parametros

def forward_propagation(X, parametros):
    caches = []
    A = X
    L = len(parametros) // 2
    for l in range(1, L):
        A_prev = A
        W, b = parametros[f'W{l}'], parametros[f'b{l}']
        Z = np.dot(W, A_prev) + b
        A = leaky_relu(Z)
        caches.append(((A_prev, W, b), Z))
    A_prev = A
    W, b = parametros[f'W{L}'], parametros[f'b{L}']
    ZL = np.dot(W, A_prev) + b
    AL = linear(ZL)
    caches.append(((A_prev, W, b), ZL))
    return AL, caches

def calcular_custo_mse(AL, Y):
    m = Y.shape[1]
    custo = (1/m) * np.sum(np.square(AL - Y))
    return custo

def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = (2/m) * (AL - Y)
    cache_atual = caches[L-1]
    (A_prev, W, b), Z = cache_atual
    dZL = linear_backward(dAL, Z)
    grads[f'dW{L}'] = np.dot(dZL, A_prev.T)
    grads[f'db{L}'] = np.sum(dZL, axis=1, keepdims=True)
    dAPrev = np.dot(W.T, dZL)
    for l in reversed(range(L-1)):
        cache_atual = caches[l]
        (A_prev, W, b), Z = cache_atual
        dZ = leaky_relu_backward(dAPrev, Z)
        grads[f'dW{l+1}'] = np.dot(dZ, A_prev.T)
        grads[f'db{l+1}'] = np.sum(dZ, axis=1, keepdims=True)
        dAPrev = np.dot(W.T, dZ)
    return grads

# ========= Otimizador Adam =========
def inicializar_adam(parametros):
    L = len(parametros) // 2; v = {}; s = {}
    for l in range(1, L + 1):
        v[f'dW{l}'] = np.zeros(parametros[f'W{l}'].shape)
        v[f'db{l}'] = np.zeros(parametros[f'b{l}'].shape)
        s[f'dW{l}'] = np.zeros(parametros[f'W{l}'].shape)
        s[f'db{l}'] = np.zeros(parametros[f'b{l}'].shape)
    return v, s

def atualizar_parametros_com_adam(parametros, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parametros) // 2; v_corrigido = {}; s_corrigido = {}
    for l in range(1, L + 1):
        v[f'dW{l}'] = beta1 * v[f'dW{l}'] + (1 - beta1) * grads[f'dW{l}']
        v[f'db{l}'] = beta1 * v[f'db{l}'] + (1 - beta1) * grads[f'db{l}']
        v_corrigido[f'dW{l}'] = v[f'dW{l}'] / (1 - beta1**t)
        v_corrigido[f'db{l}'] = v[f'db{l}'] / (1 - beta1**t)
        s[f'dW{l}'] = beta2 * s[f'dW{l}'] + (1 - beta2) * (grads[f'dW{l}']**2)
        s[f'db{l}'] = beta2 * s[f'db{l}'] + (1 - beta2) * (grads[f'db{l}']**2)
        s_corrigido[f'dW{l}'] = s[f'dW{l}'] / (1 - beta2**t)
        s_corrigido[f'db{l}'] = s[f'db{l}'] / (1 - beta2**t)
        parametros[f'W{l}'] -= learning_rate * v_corrigido[f'dW{l}'] / (np.sqrt(s_corrigido[f'dW{l}']) + epsilon)
        parametros[f'b{l}'] -= learning_rate * v_corrigido[f'db{l}'] / (np.sqrt(s_corrigido[f'db{l}']) + epsilon)
    return parametros, v, s

# ========= Modelo da Rede Neural =========
def treinar_modelo_nn(X_data, Y_data, camadas_dims, epochs, learning_rate=0.001, batch_size=32, print_custo_cada=100):
    parametros = inicializar_parametros(camadas_dims)
    v, s = inicializar_adam(parametros)
    custos = []
    t = 0
    num_amostras = X_data.shape[1]
    
    for i in range(epochs):
        permutacao = list(np.random.permutation(num_amostras))
        X_shuffled = X_data[:, permutacao]
        Y_shuffled = Y_data[:, permutacao]
        
        num_minibatches = math.floor(num_amostras / batch_size)
        for k in range(num_minibatches):
            inicio = k * batch_size
            fim = inicio + batch_size
            minibatch_X = X_shuffled[:, inicio:fim]
            minibatch_Y = Y_shuffled[:, inicio:fim]
            
            t += 1
            
            AL, caches = forward_propagation(minibatch_X, parametros)
            
            grads = backward_propagation(AL, minibatch_Y, caches)
            
            parametros, v, s = atualizar_parametros_com_adam(parametros, grads, v, s, t, learning_rate)
            
        if i % print_custo_cada == 0 or i == epochs - 1:
            custo_total = calcular_custo_mse(forward_propagation(X_data, parametros)[0], Y_data)
            custos.append(custo_total)
            print(f"Custo após época {i}: {custo_total:.8f}")
            
    return parametros, custos


# ========= FUNÇÕES DE PREVISÃO ITERATIVA =========

def encontrar_melhor_palpite(pontos_x_conhecidos, pontos_y_conhecidos, catalogo_curvas):
    melhor_erro = float('inf')
    melhor_idx = -1
    for i, params_curva in enumerate(catalogo_curvas):
        a, b, c = params_curva['a'], params_curva['b'], params_curva['c']
        y_palpite_segmento = a * pontos_x_conhecidos**2 + b * pontos_x_conhecidos + c
        erro = np.mean((pontos_y_conhecidos - y_palpite_segmento)**2)
        if erro < melhor_erro:
            melhor_erro, melhor_idx = erro, i
    return melhor_idx

def previsao_iterativa(catalogo_curvas, r1, r2, dados_radiacao_reais_selecionados):
    # dados_radiacao_reais_selecionados agora contém apenas os pontos relevantes
    num_curvas = len(catalogo_curvas)
    r_min, r_max = min(r1, r2), max(r1, r2)
    
    y_real_completa_original = dados_radiacao_reais_selecionados # Já é um array float
    
    num_pontos_reais_selecionados = len(y_real_completa_original)
    if num_pontos_reais_selecionados == 0:
        print("Erro: Não há pontos de radiação válidos para a curva secreta.")
        return # Sai se não houver dados válidos

    # Criando os pontos X para a curva real baseados nos índices dos dados selecionados
    # Mapear os índices para o intervalo [r_min, r_max] para o ajuste e plotagem.
    x_indices_selecionados = np.linspace(0, num_pontos_reais_selecionados - 1, num_pontos_reais_selecionados)
    x_plot_completo = np.interp(x_indices_selecionados, (x_indices_selecionados.min(), x_indices_selecionados.max()), (r_min, r_max))
    
    # Ajustar uma parábola aos dados de radiação selecionados
    # A curva secreta é essa parábola ajustada
    a_user, b_user, c_user = np.polyfit(x_plot_completo, y_real_completa_original, 2)
    print(f"\nCurva secreta (ajustada aos dados de Radiação): a={a_user:.4f}, b={b_user:.4f}, c={c_user:.4f}")
    
    if abs(a_user) < 1e-6:
        print("Aviso: O coeficiente 'a' da curva secreta é muito próximo de zero. A curva real será quase uma linha.")
    
    x_conhecido_max = r_min
    idx_palpite_atual = np.random.randint(0, num_curvas)
    print(f"\nSem dados iniciais, a rede fez um palpite ALEATÓRIO: Curva {idx_palpite_atual}.")

    while True:
        try:
            prompt = f"\nDigite um valor de X entre {x_conhecido_max:.2f} e {r_max:.2f} para revelar mais da curva (ou 'sair'): "
            x_user_str = input(prompt)
            if x_user_str.lower() == 'sair':
                print("Simulação encerrada pelo usuário.")
                break
            
            x_user = float(x_user_str)
            if not (x_conhecido_max < x_user <= r_max):
                print(f"Erro: O valor de X deve ser maior que {x_conhecido_max:.2f} e menor ou igual a {r_max:.2f}.")
                continue

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle(f'Análise após revelar a curva real até X = {x_user:.2f}', fontsize=16)

            # A parábola que representa a curva real ajustada
            y_real_completa_plot = a_user * x_plot_completo**2 + b_user * x_plot_completo + c_user
            
            # Dados da curva do palpite atual da rede
            params_palpite = catalogo_curvas[idx_palpite_atual]
            a_p, b_p, c_p = params_palpite['a'], params_palpite['b'], params_palpite['c']
            y_palpite_completo = a_p * x_plot_completo**2 + b_p * x_plot_completo + c_p
            
            # Definir limites de Y para manter a escala consistente
            all_y_values = np.concatenate((y_real_completa_plot, y_palpite_completo))
            y_min_global = np.min(all_y_values)
            y_max_global = np.max(all_y_values)
            y_range = y_max_global - y_min_global
            
            # Adiciona uma margem aos limites Y e lida com casos de range zero
            y_lim = (y_min_global - 0.1 * y_range, y_max_global + 0.1 * y_range) if y_range > 0 else (y_min_global - 100, y_max_global + 100)
            # Caso os dados estejam todos zerados ou muito próximos de zero, garanta uma visualização com faixa
            if y_range == 0 and y_min_global == 0:
                y_lim = (-100, 1500) # Exemplo: uma faixa mais genérica se tudo for zero
            elif y_range == 0: # Se for constante mas não zero
                y_lim = (y_min_global * 0.9, y_max_global * 1.1)


            # GRÁFICO 1: A PREVISÃO ATUAL DA REDE
            ax1.plot(x_plot_completo, y_palpite_completo, 'm--', label=f'Palpite Completo (Curva {idx_palpite_atual})')
            x_conhecido_plot_segmento = np.linspace(r_min, x_user, 200)
            y_conhecido_plot_segmento = a_user * x_conhecido_plot_segmento**2 + b_user * x_conhecido_plot_segmento + c_user
            ax1.plot(x_conhecido_plot_segmento, y_conhecido_plot_segmento, 'b-', linewidth=2.5, label='Dados Reais Já Revelados')
            ax1.set_title('1. Previsão Atual da Rede vs. Realidade')
            ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.grid(True); ax1.legend(); ax1.set_ylim(y_lim)

            # GRÁFICO 2: O GABARITO (CURVA REAL COMPLETA)
            ax2.plot(x_plot_completo, y_real_completa_plot, 'k-')
            ax2.set_title('2. A Curva Real Completa (O Gabarito)')
            ax2.set_xlabel('x'); ax2.grid(True); ax2.set_ylim(y_lim)

            # GRÁFICO 3: O PROGRESSO REAL REVELADO
            ax3.plot(x_conhecido_plot_segmento, y_conhecido_plot_segmento, 'b-', linewidth=2.5)
            ax3.set_title('3. Onde a Curva Real Está Agora')
            ax3.set_xlabel('x'); ax3.grid(True); ax3.set_ylim(y_lim)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            plt.show()

            x_conhecido_max = x_user
            pontos_x_conhecidos = np.linspace(r_min, x_conhecido_max, 200)
            pontos_y_conhecidos = a_user * pontos_x_conhecidos**2 + b_user * pontos_x_conhecidos + c_user
            
            novo_melhor_idx = encontrar_melhor_palpite(pontos_x_conhecidos, pontos_y_conhecidos, catalogo_curvas)

            if novo_melhor_idx != idx_palpite_atual:
                print(f"\nCom os dados até X={x_conhecido_max:.2f}, a rede REFINOU seu palpite.")
                print(f"Palpite anterior: {idx_palpite_atual} -> NOVO MELHOR PALPITE: {novo_melhor_idx}")
                idx_palpite_atual = novo_melhor_idx
            else:
                print(f"\nCom os dados até X={x_conhecido_max:.2f}, a rede MANTEVE seu palpite: {idx_palpite_atual}.")

            if x_conhecido_max >= r_max:
                print("\nCurva totalmente revelada! Fim da simulação."); break

        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")
        except Exception as e:
            print(f"Ocorreu um erro: {e}")

# ========= Execução Principal (Modificada para ler arquivo e fixar raízes) =========
def main_nova_rede():
    r1_input = 360
    r2_input = 1080

    print(f"\nRaízes da base de conhecimento da rede: r1={r1_input}, r2={r2_input}")

    # Definir NUM_CURVAS antes do bloco try/except
    NUM_CURVAS = 365 # Número de curvas no catálogo da rede neural

    try:
        df = pd.read_csv('Radiação-20190110.csv')
        
        # Converte a coluna 'Radiação', substituindo vírgulas por pontos e tratando erros
        df['Radiação'] = pd.to_numeric(df['Radiação'].str.replace(',', '.', regex=False), errors='coerce')
        
        # Remove linhas que resultaram em NaN na coluna 'Radiação' após a conversão
        df.dropna(subset=['Radiação'], inplace=True)

        # --- NOVA SEÇÃO DE VISUALIZAÇÃO DOS DADOS BRUTOS ---
        # Plotar os dados brutos de radiação para ajudar na seleção do intervalo
        plt.figure(figsize=(15, 6))
        plt.plot(df['Radiação'].values, label='Radiação Bruta do CSV')
        plt.title('Valores de Radiação Brutos ao Longo do Tempo (Índice da Linha)')
        plt.xlabel('Índice da Linha no CSV')
        plt.ylabel('Radiação')
        plt.grid(True)
        plt.legend()
        print("\nExibindo o gráfico da radiação bruta do CSV. Use este gráfico para identificar a faixa de índices que formam uma parábola. Feche-o para continuar.")
        plt.show()

        # --- AJUSTE MANUAL AQUI ---
        # Com base no gráfico acima, defina o índice inicial e final do período que você quer usar
        # para a curva real (onde a radiação se parece com uma parábola).
        # Exemplo: Se o gráfico mostra uma parábola clara entre os índices 200 e 1000, defina:
        start_index_for_fit = 270 # Onde a radiação começa a subir significativamente
        end_index_for_fit = 1100 # Onde a radiação começa a cair significativamente ou termina o dia

        # Garante que os índices estejam dentro dos limites do DataFrame
        start_index_for_fit = max(0, start_index_for_fit)
        end_index_for_fit = min(len(df) - 1, end_index_for_fit)

        # Seleciona a porção dos dados de radiação para a "curva secreta"
        dados_radiacao_selecionados = df['Radiação'].iloc[start_index_for_fit : end_index_for_fit + 1].values

        if len(dados_radiacao_selecionados) == 0:
            print("Erro: A porção selecionada de radiação está vazia. Verifique os índices start_index_for_fit e end_index_for_fit.")
            return
        
        # Se os dados selecionados ainda tiverem muitos zeros no início/fim,
        # você pode refinar o start_index_for_fit e end_index_for_fit com base nos dados.
        # Por exemplo, encontrar o primeiro e último índice onde a radiação é > 0.01

        # Filtra valores que são zero ou muito próximos de zero NA PORÇÃO SELECIONADA
        # Isso é importante para que o polyfit não seja distorcido por zeros que deveriam ser ignorados na forma parabólica.
        dados_radiacao_reais = dados_radiacao_selecionados[dados_radiacao_selecionados > 0.01]

        if len(dados_radiacao_reais) < 3: # Mínimo de 3 pontos para ajustar uma parábola
            print(f"Erro: Poucos pontos de radiação significativos ({len(dados_radiacao_reais)}) encontrados na faixa selecionada ({start_index_for_fit}-{end_index_for_fit}).")
            print("Ajuste os índices start_index_for_fit e end_index_for_fit para uma região com mais dados parabólicos.")
            return
        
        print(f"Dados de radiação **selecionados e filtrados** para a curva secreta: {len(dados_radiacao_reais)} pontos utilizados.")
        print(f"Valores Mínimo e Máximo em 'dados_radiacao_reais' (para curva secreta): {np.min(dados_radiacao_reais):.4f}, {np.max(dados_radiacao_reais):.4f}")

    except FileNotFoundError:
        print("Erro: O arquivo 'Radiação-20190110.csv' não foi encontrado. Certifique-se de que ele está no mesmo diretório.")
        return
    except Exception as e:
        print(f"Erro ao ler ou processar o arquivo CSV: {e}")
        # Se ocorrer um erro aqui, é bom imprimir a exceção completa para depuração
        import traceback
        traceback.print_exc()
        return

    # Ajusta Y_V_MIN e Y_V_MAX para a faixa de valores da RADIAÇÃO SELECIONADA.
    # Isso torna o catálogo de curvas da rede mais relevante para a curva real.
    Y_V_MIN = np.min(dados_radiacao_reais) * 0.8 # Multiplicado por 0.8 para dar uma margem um pouco abaixo
    Y_V_MAX = np.max(dados_radiacao_reais) * 1.2 # Multiplicado por 1.2 para dar uma margem um pouco acima

    # Lida com o caso em que a faixa de radiação selecionada é muito pequena ou constante
    if Y_V_MAX - Y_V_MIN < 100: # Se a amplitude for menor que 100 (ajuste conforme a escala dos seus dados)
        if np.max(dados_radiacao_reais) > 500: # Se os valores são grandes, dê uma faixa maior
            Y_V_MIN = np.max([0, Y_V_MIN - 200]) # Garante que não vá abaixo de zero
            Y_V_MAX = Y_V_MAX + 200
        else: # Se os valores são pequenos, dê uma faixa padrão para o catálogo
            Y_V_MIN = np.max([0, Y_V_MIN - 50])
            Y_V_MAX = Y_V_MAX + 50


    CAMADAS_DIMS = [NUM_CURVAS, 64, 32, 1]
    EPOCHS = 10000
    LEARNING_RATE = 0.0075
    BATCH_SIZE = 32
    PRINT_CUSTO_CADA = EPOCHS // 10 
    
    print("\n--- Iniciando Treinamento da Rede Neural ---")
    print(f"A rede está aprendendo seu 'catálogo' de {NUM_CURVAS} curvas com Y_v entre {Y_V_MIN:.2f} e {Y_V_MAX:.2f}...")
    
    X_one_hot, Y_vertices_reais, params_reais_todas_curvas = gerar_dados_curvas_one_hot(NUM_CURVAS, r1_input, r2_input, Y_V_MIN, Y_V_MAX)
    Y_norm, _, _ = normalizar_dados(Y_vertices_reais)
    
    parametros_treinados, custos = treinar_modelo_nn(X_one_hot, Y_norm, CAMADAS_DIMS, EPOCHS, LEARNING_RATE, BATCH_SIZE, PRINT_CUSTO_CADA)
    print("\n--- Treinamento Concluído ---")
    
    plt.figure(figsize=(10, 6))
    epochs_plot = np.linspace(0, EPOCHS, len(custos), dtype=int)
    plt.plot(epochs_plot, custos)
    plt.title("Custo do Treinamento"); plt.xlabel("Épocas"); plt.ylabel("Custo (MSE)"); plt.grid(True); plt.yscale('log')
    print("\nExibindo o gráfico do erro de treinamento. Feche-o para iniciar a simulação interativa.")
    plt.show()

    # Passa os dados de radiação reais selecionados para a função de previsão iterativa
    previsao_iterativa(params_reais_todas_curvas, r1_input, r2_input, dados_radiacao_reais)


if __name__ == "__main__":
    main_nova_rede()
