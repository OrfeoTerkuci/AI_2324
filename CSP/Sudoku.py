from typing import Set, Dict

from CSP import CSP, Variable, Value


class Sudoku(CSP):
    def __init__(self, MRV=True, LCV=True):
        super().__init__(MRV=MRV, LCV=LCV)
        self._variables = set(Cell(row, col) for row in range(9) for col in range(9))

    @property
    def variables(self) -> Set['Cell']:
        """ Return the set of variables in this CSP. """
        return self._variables

    def getCell(self, x: int, y: int) -> 'Cell':
        """ Get the variable corresponding to the cell on (x, y) """
        for cell in self.variables:
            if cell.row == y and cell.col == x:
                return cell

    def neighbors(self, var: 'Cell') -> Set['Cell']:
        """ Return all variables related to var by some constraint. """
        neighbors = set()
        for cell in self.variables:
            if (cell.row == var.row or cell.col == var.col or
                    (cell.row // 3 == var.row // 3 and cell.col // 3 == var.col // 3)):
                neighbors.add(cell)
        return neighbors - {var}

    def isValidPairwise(self, var1: 'Cell', val1: Value, var2: 'Cell', val2: Value) -> bool:
        """ Return whether this pairwise assignment is valid with the constraints of the csp. """
        return var1 not in self.neighbors(var2) if val1 == val2 else True

    def assignmentToStr(self, assignment: Dict['Cell', Value]) -> str:
        """ Formats the assignment of variables for this CSP into a string. """
        s = ""
        for y in range(9):
            if y != 0 and y % 3 == 0:
                s += "---+---+---\n"
            for x in range(9):
                if x != 0 and x % 3 == 0:
                    s += '|'

                cell = self.getCell(x, y)
                s += str(assignment.get(cell, ' '))
            s += "\n"
        return s

    def parseAssignment(self, path: str) -> Dict['Cell', Value]:
        """ Gives an initial assignment for a Sudoku board from file. """
        initialAssignment = dict()

        with open(path, "r") as file:
            for y, line in enumerate(file.readlines()):
                if line.isspace():
                    continue
                assert y < 9, "Too many rows in sudoku"

                for x, char in enumerate(line):
                    if char.isspace():
                        continue

                    assert x < 9, "Too many columns in sudoku"

                    var = self.getCell(x, y)
                    val = int(char)

                    if val == 0:
                        continue

                    assert val > 0 and val < 10, f"Impossible value in grid"
                    initialAssignment[var] = val
        return initialAssignment


class Cell(Variable):
    def __init__(self, row, col):
        super().__init__()
        self.row = row
        self.col = col
        # You can add parameters as well.

    @property
    def startDomain(self) -> Set[Value]:
        """ Returns the set of initial values of this variable (not taking constraints into account). """
        return set(range(1, 10))
