import os

import numpy as np
import json


def generate(n=10):
    # Call solver.py 100 times
    for i in range(n):
        os.system("python solver.py queens --n 10")


def calculate():
    calls_number = json.load(open("function_calls.json", "r")).values()
    calls_number = list(calls_number)

    # Calculate the mean of the calls_number list
    mean = np.mean(calls_number)
    print(f"Mean: {mean}")
    # Calculate the standard deviation of the calls_number list
    std = np.std(calls_number)
    print(f"Standard deviation: {std}")


if __name__ == "__main__":
    # generate(100)
    calculate()
