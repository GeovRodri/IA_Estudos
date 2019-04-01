#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

apes = pd.read_csv('apes-aluguel.csv')
type(apes)

plt.plot(apes.tamanho, apes.preco, 'o')
plt.show()

sb.pairplot(apes)

X = np.array(apes.tamanho)
y = np.array(apes.preco)
a = 0.01
w0 = 0.1
w1 = 0.1
epoch = 800


def hip(x, w0, w1):
    return w0 + w1*x


def plot_line(X, y, w0, w1):
    x_values = [i for i in range(int(min(X))-1,int(max(X))+2)]
    y_values = [hip(x, w0, w1) for x in x_values]
    plt.xlabel("Tamanho(m2)")
    plt.ylabel("Preço")
    plt.plot(x_values, y_values, 'r')
    plt.plot(X, y, 'o')


def media_erro(X, y, w0, w1):
    custo = 0
    m = float(len(X))
    for i in range(0, len(X)):
        custo += (hip(X[i], w0, w1) - y[i]) ** 2

    return custo / m


def gradient_descent_step(w0, w1, X, y, a):
    erro_w0 = 0
    erro_w1 = 0
    m = float(len(X))

    for i in range(0, len(X)):
        erro_w0 += hip(X[i], w0, w1) - y[i]
        erro_w1 += (hip(X[i], w0, w1) - y[i]) * X[i]

    print(erro_w0)
    novo_w0 = w0 - a * (1 / m) * erro_w0
    novo_w1 = w1 - a * (1 / m) * erro_w1

    return novo_w0, novo_w1


def gradient_descent(w0, w1, X, y, a, epoch):
    custo = np.zeros(epoch)
    for i in range(epoch):
        w0 , w1 = gradient_descent_step(w0, w1, X, y , a)
        custo[i] = media_erro(X,y,w0,w1)
        
    return w0, w1, custo


plot_line(X, y, w0, w1)
media_erro(X, y, w0, w1)
w0, w1 = gradient_descent_step(w0, w1, X, y, a)
print("w0={}, w1={}".format(w0, w1))

w0, w1, custo = gradient_descent(w0, w1, X, y, a, epoch)

# ## Plotando o custo
fig, ax = plt.subplots()  
ax.plot(np.arange(epoch), custo, 'r')  
ax.set_xlabel('Iterações')  
ax.set_ylabel('Custo')  

# ## Plotando a reta otimizada
plot_line(X, y, w0, w1)


# ## Realizando uma previsão
hip(100, w0, w1)
