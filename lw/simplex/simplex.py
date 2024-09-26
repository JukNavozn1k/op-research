import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def simplex(c, A, b):
    """
    Solve the linear program:
    Maximize:    c^T * x
    Subject to:  A * x <= b, x >= 0
    Args:
        c : Coefficients of the objective function (maximize)
        A : Constraint matrix
        b : Constraint bounds
    Returns:
        Tuple: Solution to the linear program (x values), optimal value
    """
    m, n = A.shape
    
    # Create tableau
    tableau = np.zeros((m + 1, n + m + 1))
    
    tableau[-1, :n] = -c

    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b


    print(tableau)
    while np.any(tableau[-1, :-1] < 0):
      
        pivot_col = np.argmin(tableau[-1, :-1])
        if np.all(tableau[:-1, pivot_col] <= 0):
            raise Exception('Решений бесконечно много')
       
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
    
    return x, optimal_value

c = np.array([3, 2])  
A = np.array([[1, -2],[-2,1] ])  
b = np.array([2,2])  

solution, optimal_value = simplex(c, A, b)
print("Optimal solution:", solution)
print("Optimal value:", optimal_value)
