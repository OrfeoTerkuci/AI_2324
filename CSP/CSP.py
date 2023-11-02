import random
import copy

from typing import Set, Dict, List, TypeVar, Optional
from abc import ABC, abstractmethod

from util import monitor


Value = TypeVar('Value')


class Variable(ABC):
    @property
    @abstractmethod
    def startDomain(self) -> Set[Value]:
        """ Returns the set of initial values of this variable (not taking constraints into account). """
        pass

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        return self


class CSP(ABC):
    def __init__(self, MRV=True, LCV=True):
        self.MRV = MRV
        self.LCV = LCV

    @property
    @abstractmethod
    def variables(self) -> Set[Variable]:
        """ Return the set of variables in this CSP.
            Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def remainingVariables(self, assignment: Dict[Variable, Value]) -> Set[Variable]:
        """ Returns the variables not yet assigned. """
        return self.variables.difference(assignment.keys())

    @abstractmethod
    def neighbors(self, var: Variable) -> Set[Variable]:
        """ Return all variables related to var by some constraint.
            Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def assignmentToStr(self, assignment: Dict[Variable, Value]) -> str:
        """ Formats the assignment of variables for this CSP into a string. """
        s = ""
        for var, val in assignment.items():
            s += f"{var} = {val}\n"
        return s

    def isComplete(self, assignment: Dict[Variable, Value]) -> bool:
        """ Return whether the assignment covers all variables.
            :param assignment: dict (Variable -> value)
        """
        # TODO: Implement CSP::isComplete (problem 1)
        if assignment == {}:
            return False
        for key, val in assignment.items():
            if val is None:
                return False
        return True

    @abstractmethod
    def isValidPairwise(self, var1: Variable, val1: Value, var2: Variable, val2: Value) -> bool:
        """ Return whether this pairwise assignment is valid with the constraints of the csp.
            Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def isValid(self, assignment: Dict[Variable, Value]) -> bool:
        """ Return whether the assignment is valid (i.e. is not in conflict with any constraints).
            You only need to take binary constraints into account.
            Hint: use `CSP::neighbors` and `CSP::isValidPairwise` to check that all binary constraints are satisfied.
            Note that constraints are symmetrical, so you don't need to check them in both directions.
        """
        # TODO: Implement CSP::isValid (problem 1)
        for var, val in assignment.items():
            for neighbor in self.neighbors(var):
                if neighbor not in assignment:
                    continue
                if not self.isValidPairwise(var, val, neighbor, assignment[neighbor]):
                    return False
        return True

    def solveBruteForce(self, initialAssignment: Dict[Variable, Value] = dict()) -> Optional[Dict[Variable, Value]]:
        """ Called to solve this CSP with brute force technique.
            Initializes the domains and calls `CSP::_solveBruteForce`. """
        domains = domainsFromAssignment(initialAssignment, self.variables)
        return self._solveBruteForce(initialAssignment, domains)

    @monitor
    def _solveBruteForce(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[Dict[Variable, Value]]:
        """ Implement the actual backtracking algorithm to brute force this CSP.
            Use `CSP::isComplete`, `CSP::isValid`, `CSP::selectVariable` and `CSP::orderDomain`.
            :return: a complete and valid assignment if one exists, None otherwise.
        """
        # TODO: Implement CSP::_solveBruteForce (problem 1)
        # Backtracking brute force algorithm
        # Recursive backtracking algorithm
        # 1. Check if assignment is complete
        # 2. Select unassigned variable
        # 3. Order domain values
        # 4. Loop through domain values
        # 5. Check if value is consistent with assignment
        # 6. If consistent, add to assignment and recurse
        # 7. If not consistent, remove from assignment
        # 8. If no value is consistent, return failure

        if self.isComplete(assignment) and len(assignment) == len(self.variables):
            return assignment
        var = self.selectVariable(assignment, domains)
        for val in self.orderDomain(assignment, domains, var):
            assignment[var] = val
            if self.isValid(assignment):
                result = self._solveBruteForce(assignment, domains)
                if result is not None:
                    return result
            del assignment[var]
        return None

    def solveForwardChecking(self, initialAssignment: Dict[Variable, Value] = dict()) -> Optional[Dict[Variable, Value]]:
        """ Called to solve this CSP with forward checking.
            Initializes the domains and calls `CSP::_solveForwardChecking`. """
        domains = domainsFromAssignment(initialAssignment, self.variables)
        for var in set(initialAssignment.keys()):
            domains = self.forwardChecking(initialAssignment, domains, var)
        return self._solveForwardChecking(initialAssignment, domains)

    @monitor
    def _solveForwardChecking(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[Dict[Variable, Value]]:
        """ Implement the actual backtracking algorithm with forward checking.
            Use `CSP::forwardChecking` and you should no longer need to check if an assignment is valid.
            :return: a complete and valid assignment if one exists, None otherwise.
        """
        # TODO: Implement CSP::_solveForwardChecking (problem 2)
        pass

    def forwardChecking(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]], variable: Variable) -> Dict[Variable, Set[Value]]:
        """ Implement the forward checking algorithm from the theory lectures.

        :param domains: current domains.
        :param assignment: current assignment.
        :param variable: The variable that was just assigned (only need to check changes).
        :return: the new domains after enforcing all constraints.
        """
        # TODO: Implement CSP::forwardChecking (problem 2)
        pass

    def selectVariable(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Variable:
        """ Implement a strategy to select the next variable to assign. """
        if not self.MRV:
            return random.choice(list(self.remainingVariables(assignment)))

        # TODO: Implement CSP::selectVariable (problem 2)

    def orderDomain(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]], var: Variable) -> List[Value]:
        """ Implement a smart ordering of the domain values. """
        if not self.LCV:
            return list(domains[var])

        # TODO: Implement CSP::orderDomain (problem 2)

    def solveAC3(self, initialAssignment: Dict[Variable, Value] = dict()) -> Optional[Dict[Variable, Value]]:
        """ Called to solve this CSP with AC3.
            Initializes domains and calls `CSP::_solveAC3`. """
        domains = domainsFromAssignment(initialAssignment, self.variables)
        for var in set(initialAssignment.keys()):
            domains = self.ac3(initialAssignment, domains, var)
        return self._solveAC3(initialAssignment, domains)

    @monitor
    def _solveAC3(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[Dict[Variable, Value]]:
        """
            Implement the actual backtracking algorithm with AC3.
            Use `CSP::ac3`.
            :return: a complete and valid assignment if one exists, None otherwise.
        """
        # TODO: Implement CSP::_solveAC3 (problem 3)
        pass

    def ac3(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]], variable: Variable) -> Dict[Variable, Set[Value]]:
        """ Implement the AC3 algorithm from the theory lectures.

        :param domains: current domains.
        :param assignment: current assignment.
        :param variable: The variable that was just assigned (only need to check changes).
        :return: the new domains ensuring arc consistency.
        """
        # TODO: Implement CSP::ac3 (problem 3)
        pass


def domainsFromAssignment(assignment: Dict[Variable, Value], variables: Set[Variable]) -> Dict[Variable, Set[Value]]:
    """ Fills in the initial domains for each variable.
        Already assigned variables only contain the given value in their domain.
    """
    domains = {v: v.startDomain for v in variables}
    for var, val in assignment.items():
        domains[var] = {val}
    return domains
