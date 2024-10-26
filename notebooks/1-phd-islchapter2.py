import numpy as np
from matplotlib.pyplot import subplots
import pandas as pd

# Creating a 1D array
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
# Adding two arrays
x + y
# Creating a 2D array (2 x 3 matrix)
x = np.array([[1, 2, 3], [4, 5, 6]])
x
# Number of rows
x.ndim
# Get documentation of np.array
# np.array?
# Number of rows and columns
x.shape
# Sum of all elements
z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
z.sum()
# Reshaping an array
z = np.array([1, 2, 3, 4, 5, 6])
print("Original array:\n", z)
z_reshaped = z.reshape(2, 3)
print("Reshaped array:\n", z_reshaped)
# Accessing elements of reshaped array
z_reshaped[0, 0]
z_reshaped[1, 1]

# Modifying reshaped array (also modifies original array)
a = np.array([1, 2, 3, 4, 5, 6])
print("Original array:\n", a)
a_reshaped = a.reshape(2, 3)
a_reshaped[0, 0] = 9
print("Modified reshaped array:\n", a_reshaped)
print("Modified original array:\n", a)
# Transposing an array
a_reshaped.T
# Creating an arrray of random numbers
b = np.random.normal(size=50)
b
# Adding noise to an array
c = b + np.random.normal(loc=50, scale=1, size=50)
c
# Descriptive statistics
c.mean()
c.std()
c.var()

# Graphics
fig, ax = subplots(figsize=(8, 8))
x = np.random.normal(size=100)
y = np.random.normal(size=100)
ax.scatter(x, y, marker="o", color="blue")
# Adding labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Scatter plot of X vs Y")
# Changing the size of the figure
fig.set_size_inches(12, 3)
# Creating a figure with multiple plots
fig, axes = subplots(nrows=2, ncols=3, figsize=(15, 5))
axes[0, 1].plot(x, y, "o", color="blue")
axes[1, 2].scatter(x, y, marker="+", color="red")
fig
# Creating a counter plot
fig, ax = subplots(figsize=(8, 8))
x = np.linspace(-np.pi, np.pi, 50)
y = x
f = np.multiply.outer(np.cos(y), 1 / (1 + x**2))
ax.contour(x, y, f, levels=45)
# Reading data from a file
Auto = pd.read_csv("../data/external/Auto.csv")
Auto.head()
Auto["horsepower"]
# Unique values of a column
np.unique(Auto["horsepower"])
# Finding missing values
Auto = pd.read_csv("../data/external/Auto.csv", na_values="?")
Auto["horsepower"].sum()
Auto.dropna().shape
Auto_clean = Auto.dropna()

Auto_clean.columns
