import pickle

# Load data from pickle file
with open('combined2/validate.pkl', 'rb') as f:
    data = pickle.load(f)

# Visualize the data
print("Data from pickle file:")
print(data)

# Save data to a .txt file
with open('test1.txt', 'w') as f:
    for item in data:
        f.write(str(item) + '\n')

print("Data saved to data.txt")
