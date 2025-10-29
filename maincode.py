import numpy as np
import matplotlib.pyplot as plt
import random

# ========= Parâmetros Interativos =========
def menu_interativo():
    while True:
        try:
            n_pontos = int(input("Digite a quantidade de pontos (mínimo 3, máximo 10000): "))
            if n_pontos < 3 or n_pontos > 10000:
                raise ValueError
            break
        except ValueError:
            print("Valor inválido. Tente novamente.")

    while True:
        try:
            epochs = int(input("Digite a quantidade de épocas de treinamento (ex: 10000): "))
            if epochs <= 0:
                raise ValueError
            break
        except ValueError:
            print("Valor inválido. Tente novamente.")

    print("\nDefina os parâmetros da parábola:")
    while True:
        try:
            a = float(input("Coeficiente a (abre ou fecha a parábola): "))
            r1 = float(input("Raiz 1 (primeiro ponto onde y = 0): "))
            r2 = float(input("Raiz 2 (segundo ponto onde y = 0): "))
            break
        except ValueError:
            print("Valores inválidos. Tente novamente.")

    # Coeficientes derivados da forma fatorada
    b = -a * (r1 + r2)
    c = a * (r1 * r2)

    print(f"\nA parábola terá a forma: y = {a:.2f}x² + ({b:.2f})x + ({c:.2f})")
    return n_pontos, epochs, a, b, c

# ========= Geração dos Dados =========
def gerar_dados_parabolicos(n_pontos, a, b, c):
    intervalo = n_pontos + n_pontos//2
    x_vals = sorted(random.sample(range(360, 1080), n_pontos))  # valores únicos
    y_vals = [a * (x ** 2) + b * x + c + random.uniform(-50, 50) for x in x_vals]
    return np.array(x_vals, dtype=np.float32), np.array(y_vals, dtype=np.float32)

# ========= Funções de Ativação =========
def swish(x, beta=1.0):
    return x * (1 / (1 + np.exp(-beta * x)))

def swish_derivative(x, beta=1.0):
    sigmoid = 1 / (1 + np.exp(-beta * x))
    return swish(x, beta) + sigmoid * (1 - swish(x, beta))

# ========= Treinamento =========
def treinar_rede(x, y, epochs, learning_rate=0.01, hidden_size=8, activation=swish, activation_deriv=swish_derivative):
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    n_amostras = len(x)

    np.random.seed(42)
    w1 = np.random.randn(1, hidden_size) * np.sqrt(2. / 1)
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, 1) * np.sqrt(2. / hidden_size)
    b2 = np.zeros((1, 1))

    errors = []

    for epoch in range(epochs):
        hidden_input = np.dot(x_norm.reshape(-1, 1), w1) + b1
        hidden = activation(hidden_input)
        output = np.dot(hidden, w2) + b2

        error = output.flatten() - y_norm
        mse = np.mean(error**2)
        errors.append(mse)

        # Backpropagation
        d_output = (error / n_amostras).reshape(-1, 1)
        d_w2 = np.dot(hidden.T, d_output)
        d_b2 = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = np.dot(d_output, w2.T) * activation_deriv(hidden_input)
        d_w1 = np.dot(x_norm.reshape(-1, 1).T, d_hidden)
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

        w1 -= learning_rate * d_w1
        b1 -= learning_rate * d_b1
        w2 -= learning_rate * d_w2
        b2 -= learning_rate * d_b2

        if epoch % (epochs // 10) == 0:
            print(f"Época {epoch}, Erro: {mse:.6f}")

    return w1, b1, w2, b2, x_mean, x_std, y_mean, y_std, errors

# ========= Predição =========
def prever(x_input, w1, b1, w2, b2, x_mean, x_std, y_mean, y_std):
    x_norm = (x_input - x_mean) / x_std
    hidden = swish(np.dot(x_norm.reshape(-1, 1), w1) + b1)
    y_pred_norm = np.dot(hidden, w2) + b2
    y_pred = y_pred_norm.flatten() * y_std + y_mean
    return y_pred

# ========= Execução Principal =========
def main():
    n_pontos, epochs, a, b, c = menu_interativo()
    x, y = gerar_dados_parabolicos(n_pontos, a, b, c)

    w1, b1, w2, b2, x_mean, x_std, y_mean, y_std, errors = treinar_rede(x, y, epochs)

    x_test = np.linspace(min(x)-5, max(x)+5, 200)
    y_pred = prever(x_test, w1, b1, w2, b2, x_mean, x_std, y_mean, y_std)

    # Plot
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color='red', s=80, label='Dados originais')
    plt.plot(x_test, y_pred, 'b-', linewidth=2.5, label='Previsão da rede')
    plt.title(f'Previsão com função Swish')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(errors)
    plt.title("Erro durante o treinamento")
    plt.xlabel("Épocas")
    plt.ylabel("MSE (log)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Avaliação
    y_train_pred = prever(x, w1, b1, w2, b2, x_mean, x_std, y_mean, y_std)
    print("\nComparação valores reais vs preditos:")
    for i in range(len(x)):
        print(f"x = {x[i]:5.1f} | y_real = {y[i]:7.2f} | y_pred = {y_train_pred[i]:7.2f} | Erro = {abs(y[i]-y_train_pred[i]):6.2f}")
    print(f"\nMAE: {np.mean(np.abs(y - y_train_pred)):.2f}")
    print(f"MSE: {np.mean((y - y_train_pred)**2):.2f}")

if __name__ == "__main__":
    main()
