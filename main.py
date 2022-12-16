import copy
import json
import random
import math
import numpy as np


def init_matrix(p, m):
    matrix = np.zeros((m+p, m))
    for i in range(p+m):
        for j in range(m):
            matrix[i][j] = random.randint(-1, 1) / 100
    matrix_2 = np.zeros((m, 1))
    for i in range(m):
        matrix_2[i][0] = random.randint(-1, 1) / 100
    matrix_3 = np.zeros((1, m))
    return matrix, matrix_2, matrix_3


def summary_error(e):
    result = 0
    for i in range(len(e)):
        result = result + e[i]
    return result


def add_context(vector, context_neurons, enter_size):
    for i in range(enter_size - 1, len(context_neurons[0]) + enter_size - 1):
        vector[0][i] = context_neurons[0][i - enter_size + 1]
    #print(vector)
    return vector


def activation_func(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = math.log(1 + math.exp(matrix[i][j]))
    return matrix


def derivative(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = 1 / 1 + math.exp(-matrix[i][j])
    return matrix


def correct_weights_2(matrix_weights_2, reference, y, z, alpha):
    return matrix_weights_2 - y.transpose()*(z-reference)*alpha


def correct_weights_1(matrix_weights_1, matrix_weights_2, reference, x, z, alpha):
    gamma = matrix_weights_2*(z-reference)
    F_der = derivative(x @ matrix_weights_1)
    F_der_gamma = gamma.transpose()*F_der
    X_F_der_gamma = x.transpose()@F_der_gamma
    delta_w_1 = X_F_der_gamma*alpha
    return matrix_weights_1 - delta_w_1


def weights_correction(sequence, error, alpha, p, L, N):
    W1, W2, context = init_matrix(p, L)
    E = [1500]
    X = np.zeros((1, p+L))
    iteration = 0
    #print(X)
    while summary_error(E) > error:
        E = []
        for step in range(L):
            for x in range(p):
                X[0][x] = sequence[step + x]
            X = add_context(X, context, L)
            #print(X.shape)
            #print(W1.shape)
            Y = activation_func(X @ W1)
            context = X @ W1
            Z = Y @ W2
            buffer = copy.deepcopy(W2)
            W2 = correct_weights_2(W2, sequence[step + p + 1], Y, Z, alpha)
            W1 = correct_weights_1(W1, buffer, sequence[step + p + 1], X, Z, alpha)
            print('Expect   ' + str(sequence[step + p + 1]))
            print('Result   ' + str(Z))
            E.append((sequence[step + p + 1] - Z[0][0]) * (sequence[step + p + 1] - Z[0][0]) / 2)
        iteration = iteration + 1
        print("Step" + str(iteration))
        print('Error   ' + str(summary_error(E)))
    np.save('weights1', W1)
    np.save('weights2', W2)


def predict(p, L, sequences, weight_1, weight_2):
    context = np.zeros((1, L))
    print(sequences)
    X = np.zeros((1, p+L))
    for step in range(L):
        for x in range(p):
            X[0][x] = sequences[step + x]
        X = add_context(X, context, L)
        #print(X)
        print()
        Y = activation_func(X @ weight_1)
        context = X @ weight_1
        Z = Y @ weight_2
        print(str(Z))


if __name__ == '__main__':
    seque1 = [1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
    seque2 = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    seque3 = [2**i for i in range(10)]
    seque4 = [i**2 for i in range(10)]
    #print(seque3)
    P = 3
    L = 4
    N = 200000
    print("1) Обучение\n2) Предсказание\n")
    menu = input()
    if menu == '1':
        weights_correction(seque1, 0.001, 0.00001, P, L, N)
    elif menu == '2':
        print("1) Периодическая\n2) Ряд Фибоначчи\n3) Показательная функция\n4) Степенная функция\n")
        submenu = input()
        if submenu == '1':
            w1 = np.load('weights1_cos.npy')
            w2 = np.load('weights2_cos.npy')
            predict(P, L, seque1, w1, w2)
        elif submenu == '2':
            w1 = np.load('weights1_fibbonachi.npy')
            w2 = np.load('weights2_fibbonachi.npy')
            predict(P, L, seque2, w1, w2)
        elif submenu == '3':
            w1 = np.load('weights1_power_of_2.npy')
            w2 = np.load('weights2_power_of_2.npy')
            #print(seque3)
            predict(P, L, seque3, w1, w2)
        elif submenu == '4':
            w1 = np.load('weights1_sqrt.npy')
            w2 = np.load('weights2_sqrt.npy')
            predict(P, L, seque4, w1, w2)
        print("Готово")
