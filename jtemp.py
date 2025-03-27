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

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y != 0:
        return x / y
    else:
        return "Cannot divide by zero"

def calculate(operation, x, y):
    operations = {
        'add': add,
        'subtract': subtract,
        'multiply': multiply,
        'divide': divide,
    }

    # Get the function based on the operation or use a default function
    selected_operation = operations.get(operation, lambda x, y: "Invalid operation")
    
    # Perform the calculation
    result = selected_operation(x, y)
    
    return result

# Example usage:
result_add = calculate('add', 5, 3)
print("Addition:", result_add)

result_multiply = calculate('multiply', 4, 6)
print("Multiplication:", result_multiply)

result_invalid = calculate('power', 2, 3)
print("Invalid Operation:", result_invalid)

