import numpy as np
import sys
import logging

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
def perceptron(X, y, gamma):
    logger.info("1) Initialize the weight vector w_1 to 0")
    # Initialize the weight vector w_1 to 0
    w = np.zeros(X.shape[1])
    num_rounds = 0
    mistakes = 0

    logger.info("2) For round t=1,2,...")
    while True:
        num_rounds += 1
        error_found = False
        logger.info("3) Iterate over all points x_i")
        for i in range(X.shape[0]):
            logger.info("4) If w_t ⋅ x_i < gamma/2 but x is +")
            if np.dot(w, X[i]) < gamma / 2 and y[i] == 1:
                logger.info("5) w_t+1 ← w_t + x_i")
                w += X[i]
                mistakes += 1
                error_found = True
                logger.info(f"6) exit round t = {num_rounds}")
                break
            logger.info("7) If w_t ⋅ x_i > − gamma / 2 but x is −")
            if np.dot(w, X[i]) > -gamma / 2 and y[i] == -1:
                logger.info("8) w_t+1 ← w_t − x_i")
                w -= X[i]
                mistakes += 1
                error_found = True
                logger.info(f"9) exit round t = {num_rounds}")
                break
        logger.info("10) If no mistakes this round, exit algorithm")
        if not error_found:
            break

    return w, mistakes

# # Assuming a margin value for gamma
# gamma = 1.0
#
# # Run Perceptron algorithm
# final_w, total_mistakes = perceptron(X, y, gamma)

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

# Brute-force optimal margin calculation
# def brute_force_optimal_margin(X, y):
#     from scipy.optimize import minimize
#
#     def objective(w):
#         margins = [y[i] * np.dot(w, X[i]) for i in range(X.shape[0])]
#         return -min(margins)
#
#     initial_w = np.zeros(X.shape[1])
#     result = minimize(objective, initial_w, method='BFGS')
#     optimal_w = result.x
#     optimal_margin = calculate_margin(optimal_w, X, y)
#     return optimal_margin, optimal_w
#
#
# optimal_margin, optimal_w = brute_force_optimal_margin(X, y)
#
# print(f"Optimal margin: {optimal_margin}")
# print(f"Optimal direction vector: {optimal_w}")

