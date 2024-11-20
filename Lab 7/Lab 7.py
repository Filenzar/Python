import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotData(X, y):
    plt.scatter(X, y, marker='x', c='r')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()

def computeCost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    sq_errors = (predictions - y) ** 2
    return (1 / (2 * m)) * np.sum(sq_errors)

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    for i in range(num_iters):
        error = X.dot(theta) - y
        theta -= (alpha / m) * X.T.dot(error)
    return theta

# Основной код
data = pd.read_csv('data1.txt', header=None)
X = data[0]
y = data[1]
m = len(y)
X = np.vstack([np.ones(m), X]).T
theta = np.zeros(2)

# Построение графика исходных данных
plotData(data[0], data[1])

# Вычисление начальной стоимости
cost = computeCost(X, y, theta)
print(f"Initial cost: {cost}")

# Настройка параметров для градиентного спуска
iterations = 1500
alpha = 0.01

# Запуск градиентного спуска
theta = gradientDescent(X, y, theta, alpha, iterations)
print(f"Theta: {theta}")

# Визуализация линейной регрессии
plt.scatter(X[:, 1], y, marker='x', c='r')
plt.plot(X[:, 1], X.dot(theta), '-')
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.legend(["Linear regression", "Training data"])
plt.show()
