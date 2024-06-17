import numpy as np

# Load the dataset
data = np.loadtxt('two_circle.txt')

X = data[:, :-1]  # Features
y = data[:, -1]  # Labels


# Perceptron Algorithm Implementation
def perceptron(X, y, gamma):
    # 1) Initialize the weight vector w_1 to 0
    w = np.zeros(X.shape[1])
    num_rounds = 0
    mistakes = 0

    while True:
        num_rounds += 1
        error_found = False
        # 2) Iterate over all points x_i
        for i in range(X.shape[0]):
            if np.dot(w, X[i]) < gamma / 2 and y[i] == 1:
                    w += X[i]
                    mistakes += 1
                    error_found = True
                    break
            elif np.dot(w, X[i]) > -gamma / 2 and y[i] == -1:
                    w -= X[i]
                    mistakes += 1
                    error_found = True
                    break
        if not error_found:
            break

    return w, mistakes


# Assuming a margin value for gamma
gamma = 1.0

# Run Perceptron algorithm
final_w, total_mistakes = perceptron(X, y, gamma)


# Calculate the margin achieved by the final direction vector
def calculate_margin(w, X, y):
    margins = []
    for i in range(X.shape[0]):
        margin = y[i] * np.dot(w, X[i]) / np.linalg.norm(w)
        margins.append(margin)
    return min(margins)


achieved_margin = calculate_margin(final_w, X, y)

# Output the results
print(f"Final direction vector: {final_w}")
print(f"Total mistakes made: {total_mistakes}")
print(f"Achieved margin: {achieved_margin}")


# Brute-force optimal margin calculation
def brute_force_optimal_margin(X, y):
    from scipy.optimize import minimize

    def objective(w):
        margins = [y[i] * np.dot(w, X[i]) for i in range(X.shape[0])]
        return -min(margins)

    initial_w = np.zeros(X.shape[1])
    result = minimize(objective, initial_w, method='BFGS')
    optimal_w = result.x
    optimal_margin = calculate_margin(optimal_w, X, y)
    return optimal_margin, optimal_w


optimal_margin, optimal_w = brute_force_optimal_margin(X, y)

print(f"Optimal margin: {optimal_margin}")
print(f"Optimal direction vector: {optimal_w}")

