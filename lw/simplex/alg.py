import numpy as np

def simplex_method(c, A, b):
 
    m, n = A.shape
    
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:-1, :-1] = np.hstack([A, np.eye(m)])
    tableau[:-1, -1] = b
    tableau[-1, :n] = c

    while True:
        reduced_costs = tableau[-1, :-1]
        if np.all(reduced_costs >= 0):
            print("Optimal solution found.")
            solution = np.zeros(n)
            for i in range(m):
                col = tableau[i, :n]
                if np.sum(col == 1) == 1 and np.sum(col) == 1:
                    index = np.where(col == 1)[0][0]
                    solution[index] = tableau[i, -1]
            return solution, tableau[-1, -1], "Optimal"
        
        # Step 2: Find the entering variable (most negative reduced cost)
        pivot_col = np.argmin(reduced_costs)
        if reduced_costs[pivot_col] >= 0:
            print("No negative reduced cost, optimal solution reached.")
            break
        
        # Step 3: Check for unboundedness (no positive ratios in pivot column)
        if np.all(tableau[:-1, pivot_col] <= 0):
            print("Unbounded solution detected.")
            return None, None, "Unbounded"
        
        # Step 4: Find the leaving variable (minimum positive ratio)
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[tableau[:-1, pivot_col] <= 0] = np.inf
        pivot_row = np.argmin(ratios)
        if np.isinf(ratios[pivot_row]):
            print("No feasible solution (infeasible problem).")
            return None, None, "Infeasible"
        
        # Perform pivot operation
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
        
        # Step 5: Check for alternative solutions (zero reduced costs in non-basic variables)
        non_basic_cols = [i for i in range(n) if not np.any(tableau[:-1, i] == 1)]
        if np.any(tableau[-1, non_basic_cols] == 0):
            print("Alternative optimal solution exists.")
            return None, None, "Alternative solution"

# Example usage
c = np.array([2, 3, 4])
A = np.array([[3, 2, 1], [2, 5, 3]])
b = np.array([10, 15])

solution, value, status = simplex_method(c, A, b)
print("Status:", status)
if solution is not None:
    print("Optimal solution:", solution)
    print("Optimal value:", value)
