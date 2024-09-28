import numpy as np

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(suppress=True, precision=2)

def simplex(c, A, b):
    m, n = A.shape
    tableau = np.zeros((m + 1, n + m + 1))
    
    tableau[-1, :n] = -c
    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b

    has_alternative = False 

    print(tableau)
    while np.any(tableau[-1, :-1] < 0):
        
        if np.any(b < 0):
            raise Exception('Задача несовместна. Есть отрицательные элементы в векторе правых частей b.')
        pivot_col = np.argmin(tableau[-1, :-1])
        if np.all(tableau[:-1, pivot_col] <= 0):
            raise Exception('Задача не ограничена. Все элементы разрешающего столбца <= 0')
       
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf
        pivot_row = np.argmin(ratios)

        tableau[pivot_row] /= tableau[pivot_row, pivot_col]
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        print(tableau)

    x = np.zeros(n)
    for i in range(n):
        column = tableau[:-1, i]
        if np.count_nonzero(column) == 1 and np.any(column == 1):
            x[i] = tableau[np.argmax(column), -1]
    
    optimal_value = tableau[-1, -1]

    for j in range(n):
        if tableau[-1, j] == 0:
            column = tableau[:-1, j]
            if np.any(column > 0) and np.count_nonzero(column) > 1:  
                has_alternative = True
                break
    
    return x, optimal_value,has_alternative

c = np.array([3,2])  
A = np.array([[-1,-2],[-2,-1] ])  
b = np.array([4,5])  

solution, optimal_value,has_alternative = simplex(c, A, b)
print("Оптимальное решение: ", solution)
print("Оптимальное значение:", optimal_value)
print('Альтернативные решения: ', has_alternative)