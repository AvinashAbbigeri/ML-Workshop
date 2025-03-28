# print("Hello World!!")

# a = int(input("Enter a number: "))
# b = int(input("Enter another number: "))
# c =a + b
# print(f"Output is: {c}")
 
# # For Loop
# for i in range(3):
#  print("Iteration:", i)

# # While Loop
# count = 1
# while count <= 3:
#   print("While Loop Count:", count)
#   count += 1

# Program to find the sum of numbers in a list
# numbers = [2, 4, 6, 8, 10]
# sum_numbers = 0
# for num in numbers:
#  sum_numbers += num
# print("Sum:", sum_numbers)

# def greet(name="Guest"):
#     print("Hello,", name)
# greet("Mohan")
# greet()

# def add(x, y):
#     return x + y

# def subtract(x, y):
#     return x - y

# def multiply(x, y):
#     return x * y

# def divide(x, y):
#     if y != 0:
#         return x / y
#     else:
#         return "Cannot divide by zero"

# def calculate(operation, x, y):
#     operations = {
#         'add': add,
#         'subtract': subtract,
#         'multiply': multiply,
#         'divide': divide,
#     }

#     # Get the function based on the operation or use a default function
#     selected_operation = operations.get(operation, lambda x, y: "Invalid operation")
    
#     # Perform the calculation
#     result = selected_operation(x, y)
    
#     return result

# # Example usage:
# result_add = calculate('add', 5, 3)
# print("Addition:", result_add)

# result_multiply = calculate('multiply', 4, 6)
# print("Multiplication:", result_multiply)

# result_invalid = calculate('power', 2, 3)
# print("Invalid Operation:", result_invalid)

# class Calculator:
#     firstNumber = 0.0
#     secondNumber = 0.0

#     def __init__(self, fn, sn):
#         self.firstNumber = fn
#         self.secondNumber = sn

#     def add(self):
#         return self.firstNumber + self.secondNumber
    
#     def sub(self):
#         return self.firstNumber - self.secondNumber
    
#     def mul(self):
#         return self.firstNumber * self.secondNumber

#     def div(self):
#         # Handling division by zero to avoid errors
#         if self.secondNumber != 0:
#             return self.firstNumber / self.secondNumber
#         else:
#             return "Error! Division by zero."

# # Taking input from the user
# fn = float(input("Enter the first number: "))
# sn = float(input("Enter the second number: "))

# # Create a Calculator object
# cal = Calculator(fn, sn)

# # Print the results
# print("Add:", cal.add())
# print("Sub:", cal.sub())
# print("Mul:", cal.mul())
# print("Div:", cal.div())

# # Creating a list
# fruits = ["Apple", "Banana", "Mango"]
# # Adding an element
# fruits.append("Orange")
# # Removing an element
# fruits.remove("Banana")
# # Accessing elements
# print(fruits[0])  # Output: Apple
# # Iterating over the list
# for fruit in fruits:
#     print(fruit)

# #Creating a set
# numbers = {1, 2, 3, 4, 4, 5}  # Duplicate '4' will be removed automatically

# # Adding elements
# numbers.add(6)

# # Removing elements
# numbers.discard(2)

# # Checking membership
# print(3 in numbers)  # Output: True

# # Iterating over the set
# for num in numbers:
#     print(num)
# # Creating a dictionary
# student = {"name": "Rahul", "age": 21, "course": "Python"}

# # Adding a new key-value pair
# student["grade"] = "A"

# # Removing a key
# student.pop("age")

# # Accessing values
# print(student["name"])  # Output: Rahul

# # Iterating over dictionary
# for key, value in student.items():
#     print(f"{key}: {value}")


# # Creating a tuple
# coordinates = (10, 20, 30)

# # Accessing elements
# print(coordinates[1])  # Output: 20

# # Tuples are immutable, so the following will cause an error:
# # coordinates[1] = 40  # TypeError: 'tuple' object does not support item assignment

# # Iterating over a tuple
# for value in coordinates:
#     print(value)

# import numpy as np  

# # Creating an array
# arr = np.array([1, 2, 3, 4, 5])  
# print("Array:", arr)

# # Basic Operations
# print("Mean:", np.mean(arr))  # Calculate mean
# print("Sum:", np.sum(arr))  # Calculate sum

# # Reshaping an array
# matrix = arr.reshape(1, 5)
# print("Reshaped Array:", matrix)

# import pandas as pd  

# # Creating a DataFrame
# data = {"Name": ["Amit", "Sara", "Raj"], "Age": [25, 30, 22]}  
# df = pd.DataFrame(data)  
# print(df)

# # Accessing a column
# print("Ages:", df["Age"])  

# # Filtering Data
# filtered_df = df[df["Age"] > 25]  
# print("Filtered Data:", filtered_df)

# from scipy import stats  
# import numpy as np  

# # Creating sample data
# data = np.array([2, 3, 5, 6, 9])  

# # Calculating statistics
# mean_value = stats.tmean(data)  
# median_value = np.median(data)  
# print("Mean:", mean_value)  
# print("Median:", median_value)

# import matplotlib.pyplot as plt

# # Data
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 15, 25, 30]

# # Plot
# plt.plot(x, y, marker='o', linestyle='-', color='blue')

# # Labels & Title
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Simple Line Plot")

# # Save Plot to a file
# plt.savefig("plot.png")

# # Show Plot (optional if in an interactive environment)
# plt.show()

# import matplotlib.pyplot as plt

# # Data
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 15, 25, 30]

# # Scatter Plot
# plt.scatter(x, y, color='red', marker='s')

# # Labels & Title
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Scatter Plot")

# plt.savefig("plot1.png")
# # Show Plot
# plt.show()

# import matplotlib.pyplot as plt

# # Data
# categories = ['A', 'B', 'C', 'D', 'E']
# values = [10, 25, 15, 30, 20]

# # Bar Plot
# plt.bar(categories, values, color='green')

# # Labels & Title
# plt.xlabel("Categories")
# plt.ylabel("Values")
# plt.title("Bar Chart Example")


# plt.savefig("plot2.png")
# # Show Plot
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Generate Random Data
# data = np.random.randn(1000)  # 1000 random numbers

# # Histogram
# plt.hist(data, bins=20, color='purple', edgecolor='black')

# # Labels & Title
# plt.xlabel("Value Range")
# plt.ylabel("Frequency")
# plt.title("Histogram Example")

# plt.savefig("plot3.png")
# # Show Plot
# plt.show()

# import matplotlib.pyplot as plt

# # Data
# labels = ['Apple', 'Banana', 'Orange', 'Grapes']
# sizes = [30, 25, 20, 25]
# colors = ['red', 'yellow', 'orange', 'purple']

# # Pie Chart
# plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

# # Title
# plt.title("Fruit Distribution")

# plt.savefig("plot4.png")
# # Show Plot
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Generate synthetic dataset
# np.random.seed(42)
# sqft = np.random.randint(800, 4000, 100)  # Square footage
# bedrooms = np.random.randint(1, 5, 100)  # Number of bedrooms
# price = (sqft * 300) + (bedrooms * 10000) + np.random.randint(20000, 50000, 100)  # House price

# df = pd.DataFrame({'SqFt': sqft, 'Bedrooms': bedrooms, 'Price': price})

# # Train-test split
# X = df[['SqFt', 'Bedrooms']]
# y = df['Price']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Evaluation Metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# # Print Evaluation Metrics
# print(f"Mean Absolute Error: {mae}")
# print(f"Mean Squared Error: {mse}")
# print(f"Root Mean Squared Error: {rmse}")
# print(f"R² Score: {r2}")

# # Plot Predictions vs Actual Prices
# plt.scatter(y_test, y_pred, color='green', alpha=0.5, label="Predicted vs Actual")

# # Regression Line (Perfect Prediction Line)
# min_val = min(y_test.min(), y_pred.min())
# max_val = max(y_test.max(), y_pred.max())
# plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Regression Line")

# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual vs Predicted House Prices")
# plt.legend()
# plt.savefig("plot5.png")
# plt.show()    

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Generate synthetic dataset
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Evaluation Metrics
silhouette_avg = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette_avg}")

# Plot Clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="x", s=200, color="red")
plt.title("Customer Segmentation using K-Means Clustering")
plt.savefig("plot6.png")
plt.show()