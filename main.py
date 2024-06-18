import numpy as np
import sys
import logging

from scipy.optimize import minimize

# Configure logging
# ! If you want to print the logger, lower the level to INFO !
logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# Load the dataset
data = np.loadtxt('two_circle.txt')

X = data[:, :-1]  # Features
# print(X)
y = data[:, -1]  # Labels
# print(y)

# Perceptron Algorithm Implementation
def perceptron(X, y):
    logger.info("1) Initialize the weight vector w_1 to 0")
    # 1. Set vector w_1 = 0
    w = np.zeros(X.shape[1])

    num_rounds = 0
    mistakes = 0

    logger.info("2) For round t=1,2,...")
    while True:
        num_rounds += 1
        error_found = False
        logger.info("3) Iterate over all points x_i")
        for i in range(X.shape[0]):
            # 4. If w_t ⋅ x_i > 0
            if np.dot(w, X[i]) > 0:
                logger.info("Guess +")
                y_pred = 1
            else:
                logger.info("Guess -")
                y_pred = -1

            # 6. On mistake:
            if y_pred == 1 and y[i] == -1:
                logger.info("On mistake: w_t+1 ← w_t - x_i")
                w -= X[i]
                mistakes += 1
                error_found = True
                logger.info(f"9) exit round t = {num_rounds}")
                break
            elif y_pred == -1 and y[i] == 1:
                logger.info("On mistake: w_t+1 ← w_t + x_i")
                w += X[i]
                mistakes += 1
                error_found = True
                logger.info(f"9) exit round t = {num_rounds}")
                break

        logger.info("10) If no mistakes this round, exit algorithm")
        if not error_found:
            break

    return w, mistakes


# Run Perceptron algorithm
final_w, total_mistakes = perceptron(X, y)

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

# Function to calculate the optimal margin using brute force
def brute_force_margin(X, y):
    best_margin = 0
    best_w = None
    search_range = 20
    steps = 100
    values = np.linspace(-search_range, search_range, num=steps)

    # Generate all possible weight vectors
    weight_vectors = np.array([[w0, w1] for w0 in values for w1 in values])

    for w in weight_vectors:
        norm_w = np.linalg.norm(w)
        if norm_w == 0:
            continue
        margins = [y[i] * np.dot(w, X[i]) / norm_w for i in range(len(X))]
        margin = min(margins)
        if margin > best_margin:
            best_margin = margin
            best_w = w

    return best_margin, best_w


# Calculate the optimal margin and direction vector using brute force
optimal_margin, optimal_w = brute_force_margin(X, y)

print(f"Optimal margin: {optimal_margin}")
print(f"Optimal direction vector: {optimal_w}")
