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
        return self.remainingVariables(assignment) == set()

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
    def _solveBruteForce(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[
        Dict[Variable, Value]]:
        """ Implement the actual backtracking algorithm to brute force this CSP.
            Use `CSP::isComplete`, `CSP::isValid`, `CSP::selectVariable` and `CSP::orderDomain`.
            :return: a complete and valid assignment if one exists, None otherwise.
        """
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

    def solveForwardChecking(self, initialAssignment: Dict[Variable, Value] = dict()) -> Optional[
        Dict[Variable, Value]]:
        """ Called to solve this CSP with forward checking.
            Initializes the domains and calls `CSP::_solveForwardChecking`. """
        domains = domainsFromAssignment(initialAssignment, self.variables)
        for var in set(initialAssignment.keys()):
            domains = self.forwardChecking(initialAssignment, domains, var)
        return self._solveForwardChecking(initialAssignment, domains)

    @monitor
    def _solveForwardChecking(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[
        Dict[Variable, Value]]:
        """ Implement the actual backtracking algorithm with forward checking.
            Use `CSP::forwardChecking` and you should no longer need to check if an assignment is valid.
            :return: a complete and valid assignment if one exists, None otherwise.
        """
        # TODO: Implement CSP::_solveForwardChecking (problem 2)
        if self.isComplete(assignment):
            return assignment
        for domain in domains.values():
            if len(domain) == 0:
                return None
        var = self.selectVariable(assignment, domains)
        for val in self.orderDomain(assignment, domains, var):
            original_domains = copy.deepcopy(domains)
            assignment[var] = val
            # Adjust domains
            domains = self.forwardChecking(assignment, domains, var)
            # Recurse next assignment
            result = self._solveForwardChecking(assignment, domains)
            if result is not None:
                return result
            # Backtrack
            del assignment[var]
            # Restore domains
            domains = original_domains
        return None

    def forwardChecking(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]],
                        variable: Variable) -> Dict[Variable, Set[Value]]:
        """ Implement the forward checking algorithm from the theory lectures.

        :param domains: current domains.
        :param assignment: current assignment.
        :param variable: The variable that was just assigned (only need to check changes).
        :return: the new domains after enforcing all constraints.
        """
        value = assignment[variable]
        for var in self.neighbors(variable):
            # new_domain =
            for val in copy.deepcopy(domains[var]):
                # Remove inconsistent values
                if not self.isValidPairwise(variable, value, var, val):
                    domains[var].remove(val)
                    if len(domains[var]) == 0:
                        # Restore original domains if inconsistency found
                        return domains
        return domains

    def selectVariable(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Variable:
        """ Implement a strategy to select the next variable to assign. """
        if not self.MRV:
            return random.choice(list(self.remainingVariables(assignment)))

        # TODO: Implement CSP::selectVariable (problem 2)

        # Selection of variable with minimum remaining values,
        # if multiple variables have the same amount of remaining values, select the one with the most constraints

        # Get the variable with the minimum remaining values
        min_remaining_values = float('inf')
        min_remaining_values_var = []
        for var in self.remainingVariables(assignment):
            if len(domains[var]) < min_remaining_values:
                min_remaining_values = len(domains[var])
                min_remaining_values_var.append(var)

        if len(min_remaining_values_var) == 1:
            return min_remaining_values_var[0]

        # Get the variable with the most constraints
        max_constraints = 0
        max_constraints_var = None
        for var in min_remaining_values_var:
            if len(self.neighbors(var)) > max_constraints:
                max_constraints = len(self.neighbors(var))
                max_constraints_var = var
        return max_constraints_var

    def orderDomain(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]], var: Variable) -> \
            List[Value]:
        """ Implement a smart ordering of the domain values. """
        if not self.LCV:
            return list(domains[var])

        # TODO: Implement CSP::orderDomain (problem 2)

        # Order the domain values by the least constraining value heuristic
        # Sort the values by the number of values they rule out for other variables

        # Get the number of values ruled out for each value
        ruled_out_values = {}
        for val in domains[var]:
            ruled_out_values[val] = 0
            for neighbor in self.neighbors(var):
                if val in domains[neighbor]:
                    ruled_out_values[val] += 1
        # Sort the values by the number of values they rule out for other variables
        return sorted(domains[var], key=lambda x: ruled_out_values[x])

    def solveAC3(self, initialAssignment: Dict[Variable, Value] = dict()) -> Optional[Dict[Variable, Value]]:
        """ Called to solve this CSP with AC3.
            Initializes domains and calls `CSP::_solveAC3`. """
        domains = domainsFromAssignment(initialAssignment, self.variables)
        for var in set(initialAssignment.keys()):
            domains = self.ac3(initialAssignment, domains, var)
        return self._solveAC3(initialAssignment, domains)

    @monitor
    def _solveAC3(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[
        Dict[Variable, Value]]:
        """
            Implement the actual backtracking algorithm with AC3.
            Use `CSP::ac3`.
            :return: a complete and valid assignment if one exists, None otherwise.
        """
        # TODO: Implement CSP::_solveAC3 (problem 3)
        pass

    def ac3(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]], variable: Variable) -> Dict[
        Variable, Set[Value]]:
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
