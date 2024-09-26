import numpy as np

def simplex(c, A, b):
    # Add slack variables to convert inequalities to equalities
    m, n = A.shape
    A = np.hstack([A, np.eye(m)])  # Add slack variables
    c = np.hstack([c, np.zeros(m)])  # Adjust the cost vector

    # Initial Basic Feasible Solution (BFS)
    basis = list(range(n, n + m))
    
    # Initialize the tableau
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n + m] = A
    tableau[:m, -1] = b
    tableau[-1, :n + m] = -c

    def pivot(tableau, row, col):
        tableau[row] /= tableau[row, col]
        for i in range(len(tableau)):
            if i != row:
                tableau[i] -= tableau[i, col] * tableau[row]

    while np.any(tableau[-1, :-1] < 0):
        # Determine entering variable (most negative coefficient in the objective row)
        col = np.argmin(tableau[-1, :-1])
        
        # Check for unboundedness (if all elements in the column are ≤ 0)
        if np.all(tableau[:-1, col] <= 0):
            return None, None, False, True  # Unbounded solution
        
        # Determine leaving variable (smallest ratio of b[i] / A[i, col])
        ratios = tableau[:-1, -1] / tableau[:-1, col]
        valid_ratios = np.where(tableau[:-1, col] > 0, ratios, np.inf)
        
        # Check if there are no valid ratios (infeasible problem)
        if np.all(valid_ratios == np.inf):
            return None, None, True, False  # No feasible solution
        
        row = np.argmin(valid_ratios)
        
        # Pivot
        pivot(tableau, row, col)
        basis[row] = col

    # Extract the solution
    solution = np.zeros(n + m)
    solution[basis] = tableau[:-1, -1]
    optimal_value = tableau[-1, -1]
    
    # Check for alternative solutions:
    alternative_solution_exists = np.any((tableau[-1, :-1] == 0) & (solution[:-m] == 0))

    return solution[:n], optimal_value, alternative_solution_exists, False

# Example Usage
c = np.array([3, 2])  # Objective function coefficients
A = np.array([[1, 2],
              [1, 1]])  # Constraint coefficients
b = np.array([4, 3])  # Right-hand side values

optimal_solution, optimal_value, has_alternative, is_unbounded = simplex(c, A, b)

if optimal_solution is None and optimal_value is None:
    print("No feasible solution exists." if not is_unbounded else "The solution is unbounded.")
else:
    print("Optimal Solution:", optimal_solution)
    print("Optimal Value:", optimal_value)
    print("Alternative Solution Exists:", has_alternative)
