""" Command line interface to call the CSP solver. """
from enum import Enum

from typer import Typer
from tqdm import tqdm

from Sudoku import Sudoku
from NQueens import NQueens


# IMPORTANT: Do not edit this file!


class Method(str, Enum):
    bf = "bf"
    fc = "fc"
    ac3 = "ac3"


app = Typer()


def solve(csp, method: Method, initialAssignment=dict()):
    if method == Method.bf:
        print("Solving with brute force")
        assignment = csp.solveBruteForce(initialAssignment)
    elif method == Method.fc:
        print("Solving with forward checking")
        assignment = csp.solveForwardChecking(initialAssignment)
    elif method == Method.ac3:
        print("Solving with ac3")
        assignment = csp.solveAC3(initialAssignment)
    else:
        raise RuntimeError(f"Method '{method}' not found.")

    if assignment:
        s = csp.assignmentToStr(assignment)
        tqdm.write("\nSolution:")
        tqdm.write(s)
    else:
        tqdm.write("No solution found")


@app.command()
def sudoku(path: str, method: Method = Method.bf, MRV: bool = True, LCV: bool = True):
    """ Solve Sudoku as a CSP. """
    if method == Method.bf:
        MRV = False
        LCV = False
    csp = Sudoku(MRV=MRV, LCV=LCV)
    initialAssignment = csp.parseAssignment(path)
    solve(csp, method, initialAssignment)


@app.command()
def queens(n: int = 5, method: Method = Method.bf, MRV: bool = True, LCV: bool = True):
    """ Solve the N Queens problem as a CSP. """
    if method == Method.bf:
        MRV = False
        LCV = False
    csp = NQueens(n=n, MRV=MRV, LCV=LCV)
    solve(csp, method)


@app.command()
def multi_queens(n: int = 5, method: Method = Method.bf, MRV: bool = True, LCV: bool = True,
                 i: int = 1, mean: bool = False, std: bool = False):
    """ Solve the N Queens problem as a CSP. """
    import json
    import numpy as np
    import os

    # Create the function_calls.json file
    open('function_calls.json', 'w').close()

    for _ in range(i):
        command = (f"python solver.py queens --n {n} "
                   f"--method {method.value} "
                   f"{'--mrv' if MRV else '--no-mrv'} "
                   f"{'--lcv' if LCV else '--no-lcv'}")
        os.system(command)

    # Load the data from the json file
    calls_number = json.load(open("function_calls.json", "r")).values()
    calls_number = list(calls_number)
    calls_number = np.array(calls_number)

    # Calculate the mean of the calls_number list
    if mean:
        mean = np.mean(calls_number)
        data = json.load(open("function_calls.json", "r"))
        data["mean"] = mean
        json.dump(data, open("function_calls.json", "w"))
        print(f"Mean: {mean}")
    # Calculate the standard deviation of the calls_number list
    if std:
        std = np.std(calls_number)
        data = json.load(open("function_calls.json", "r"))
        data["std"] = std
        json.dump(data, open("function_calls.json", "w"))
        print(f"Standard deviation: {std}")

    print(f"Max: {max(calls_number)}")
    print(f"Min: {min(calls_number)}")


if __name__ == "__main__":
    app()
