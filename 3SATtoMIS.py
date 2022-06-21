#!/usr/bin/env python3

from random import randint
from copy import deepcopy
from networkx.algorithms.approximation import maximum_independent_set

import networkx as nx
import matplotlib.pyplot as plt

"""
    Generates random 3SAT problems and reduces them to an independent set problem 
    as described in Algorithm Design by Jon Klenberg and Eva Tardos
    TODO:   Option to search for and only display unsatisfiable examples
            Allow for user provided 3SAT problems
            Code cleanup...
"""

class ThreeSATLiteral:

    """
        Class representing literal objects
        Boolean values that may be negated
        Each object has an ID, value (True/False) and a negated flag
    """

    def __init__(self, literal_id):

        self.literal_id = literal_id
        self.value = bool(randint(0, 1))
        self.negated = False

    def __str__(self):

        if self.negated:
            return f"~x{self.literal_id}"
        else:
            return f"x{self.literal_id}"

    def __int__(self):

        return self.literal_id

    def __bool__(self):

        return self.value

    def string_with_value(self):

        if self.negated:
            return f"~x{self.literal_id}({self.value})"
        else:
            return f"x{self.literal_id}({self.value})"

    def negate(self):

        """
            returns a deep copy of the literal
            value is set to the opposet of the original value
            negated flag is set to True
        """

        negated_copy = deepcopy(self)
        negated_copy.negated = True
        negated_copy.value = not self.value

        return negated_copy


class ThreeSATClause:

    """
        a clause containing three literals
        the result of the clause is true if any of the literals are true
    """

    def __init__(self, literals: list):

        self.literals = deepcopy(literals)

        self.result = bool(self.literals[0] or self.literals[1] or self.literals[2])

        self.edge_list = [
            [self.literals[0], self.literals[1]],
            [self.literals[1], self.literals[2]],
            [self.literals[0], self.literals[2]],
        ]

    def __str__(self):

        return f"({self.literals[0]} v {self.literals[1]} v {self.literals[2]})"

    def __bool__(self):

        return self.result


class ThreeSAT:

    """
        contains a SAT with an arbatrary number of clauses where each clause contains 3 literals
    """


    def __init__(self, size=3, num_literals = 0):

        if num_literals == 0:
            num_literals = size*3

        # ensure consistancy across values by creating literals here and using deep copies
        literals = [ThreeSATLiteral(i) for i in range(0, num_literals)]
        literal_locations = [[] for _ in range(num_literals)]
        negated_literal_locations = [[] for _ in range(num_literals)]

        self.clause_list = []

        # randomly select literals for use in clauses
        for i in range(size):

            clause_literal_ids = []
            clause_literals = []

            for j in range(3):

                while True:

                    literal_id = randint(0, num_literals - 1)
                    negated = bool(randint(0, 1))
                    literal = literals[literal_id]

                    if literal_id not in clause_literal_ids:

                        if negated:

                            literal = literal.negate()
                            negated_literal_locations[literal_id].append([i, j])

                        else:
                            literal_locations[literal_id].append([i, j])

                        clause_literal_ids.append(literal_id)
                        clause_literals.append(literal)
                        break

            self.clause_list.append(ThreeSATClause(clause_literals))

        self.edge_list = []

        # build the edge list by comparing the list of literals to negated literals
        for i in range(num_literals):

            literal_count = len(literal_locations[i])
            negated_literal_count = len(negated_literal_locations[i])

            if literal_count > 0 and negated_literal_count > 0:

                for literal_location in literal_locations[i]:

                    for negated_location in negated_literal_locations[i]:

                        literal = self.clause_list[literal_location[0]].literals[
                            literal_location[1]
                        ]

                        negated_literal = self.clause_list[
                            negated_location[0]
                        ].literals[negated_location[1]]

                        self.edge_list.append([literal, negated_literal])

        self.size = size
        self.result = all(self.clause_list)

    def __str__(self):

        string = ""

        for i in range(0, self.size):

            if i < self.size - 1:
                string += f"{self.clause_list[i]} ^ "
            else:
                string += f"{self.clause_list[i]}"

        return string

    def __iter__(self):

        self.n = -1
        return self

    def __next__(self):

        if self.n < len(self.clause_list) - 1:

            self.n += 1
            return self.clause_list[self.n]

        else:
            raise StopIteration


    def _gen_random_clause(self, num_literals):

        # TODO move the current logic here and make the default user provided clauses
        pass


def graph_three_sat(three_sat, find_unsat):

    graph = nx.Graph()

    for sat in three_sat:

        graph.add_edges_from(sat.edge_list)

    graph.add_edges_from(three_sat.edge_list)
    pos = nx.shell_layout(graph, scale=0.4)
    k = maximum_independent_set(graph)

    color_map = []

    for node in graph:
        if node in k:
            color_map.append("cyan")
        else:
            color_map.append("gray")

    satisfiable = len(k) >= three_sat.size

    if find_unsat and satisfiable:

        return

    nx.draw_networkx(graph, font_size=8, node_color=color_map, pos=pos)
    plt.title(
        f"{three_sat}\nk={len(k)} size={three_sat.size}\nSatisfiable: {satisfiable}",
        size=8,
    )
    plt.show()


if __name__ == "__main__":

    size = int(input("size: "))

    num_literals = 0
    max_literal_count = size * 3

    while num_literals < 3 or num_literals > max_literal_count:

        try:

            num_literals = int(input(f"number of unique literals (3-{size*3}): "))

        except TypeError:

            pass

    find_unsat = input("Find first unsatisfiable? [Y/n] ").lower()

    try:

        if find_unsat[0] == "y":
            find_unsat = True
        else:
            find_unsat = False

    except IndexError:

        find_unsat = True


    while True:
        three_sat = ThreeSAT(size=size, num_literals=num_literals)
        print(three_sat)
        graph_three_sat(three_sat, find_unsat)