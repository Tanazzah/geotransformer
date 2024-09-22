import numpy as np

def normalize_matrix(matrix):
    max_value = np.max(matrix)
    print(matrix)
    return matrix / max_value

def matrix_difference_percentage(matrix1, matrix2):
    # Normalize matrices
    norm_matrix1 = normalize_matrix(matrix1)
    norm_matrix2 = normalize_matrix(matrix2)
    
    # Compute absolute element-wise difference
    abs_diff = np.abs(norm_matrix1 - norm_matrix2)
    
    # Compute mean absolute difference
    mean_abs_diff = np.mean(abs_diff)
    
    # Compute percentage difference
    percentage_diff = mean_abs_diff * 100
    
    return percentage_diff

def read_first_matrix_from_file(filename):
    with open(filename, 'r') as file:
        matrix_lines = []
        for line in file:
            if line.strip():  # Non-empty line
                row = [float(num) for num in line.split()]
                matrix_lines.append(row)
            else:  # Empty line indicates the end of a matrix
                return np.array(matrix_lines)
    return np.array(matrix_lines)  # Return the first matrix if the file doesn't contain an empty line

# Read the first matrix from file A
matrix_A = read_first_matrix_from_file("input_matrices.txt")

# Read the first matrix from file B
matrix_B = read_first_matrix_from_file("transform_matrices.txt")

# Compute and print the percentage difference
percentage_diff = matrix_difference_percentage(matrix_A, matrix_B)
print("Percentage difference for the first matrix:", percentage_diff)

