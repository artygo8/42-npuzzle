#!/bin/python3
import random
import sys

GREEDY = 0
INFINITY = 10 ** 10
DEFAULT_SIZE = 4
HEURISTIC = 0  # Choose from 0: Pythagore, 1: Manhattan, 2: ???


class dir:
    UP, RIGHT, DOWN, LEFT = range(4)


class NPuzzle:

    recorded_final_lsts = dict()  # size: list

    # Low Level Functions

    def __init__(self, size=0, lst=False, local=INFINITY, filename=None) -> None:
        self.size = size

        if filename:
            lst = self.parse(filename)

        if self.size not in self.recorded_final_lsts:
            self.recorded_final_lsts[self.size] = self.final_lst()

        self.lst = lst or self.random_lst()
        self.sign = " ".join(map(str, self.lst))
        self.parent_sign = None
        self.local = local if lst and not filename else 0
        self.glob = self.greedy() if GREEDY else self.heuristic()

    def __eq__(self, other) -> bool:
        return self.sign == other.sign

    def __str__(self) -> str:
        return "\n".join(
            " ".join(
                f"{number:{len(str(self.size**2))+1}d}"
                for number in self.lst[i * self.size : (i + 1) * self.size]
            )
            for i in range(self.size)
        )

    def copy_goals(self, other) -> None:
        self.parent_sign = other.parent_sign
        self.local = other.local
        self.glob = other.glob

    def find(self, n: int) -> (int, int):
        index = self.lst.index(n)
        return (index % self.size, index // self.size)

    def parse(self, filename) -> list:

        self.size = 0
        lst = []

        with open(filename, "r") as f:

            for line in f.readlines():

                line = line.split("#")[0]
                current_size = 0

                for enum, elem in enumerate(line.strip().split()):
                    try:
                        if self.size == 0:
                            self.size = int(elem)
                        else:
                            lst.append(int(elem))
                    except Exception:
                        raise Exception("What the fuck, mate ? These ain't numbers !")
                    current_size = enum + 1

                if current_size != self.size and current_size != 1:
                    raise Exception("Incoherent sizes")

        if sorted(lst) != [*range(self.size ** 2)]:
            raise Exception("This file contains bad numbers")

        return lst

    # Tool Functions

    def random_lst(self) -> list:
        return random.sample(range(self.size ** 2), self.size ** 2)

    def final_lst(self) -> list:
        return [
            (j + 1, 0)[j + 1 == self.size ** 2]
            for j in [
                self.to_snake_ordered(self.size).index(i) for i in range(self.size ** 2)
            ]
        ]

    def count_inversions(self, lst) -> int:
        tmp = [i for i in lst if i != 0]
        return sum(
            tmp[i] > tmp[j] for i in range(len(tmp)) for j in range(i + 1, len(tmp))
        )

    def solvable(self) -> bool:
        return not self.count_inversions(self.to_snake()) % 2

    def to_snake(self) -> list:
        return [self.lst[index] for index in self.to_snake_ordered(self.size)]

    def to_snake_ordered(self, size) -> list:
        """gives the spirale order for an ordered grid.

        Example:
            --> size = 4

                               0  1  2  3
            --> ordered grid:  4  5  6  7
                               8  9 10 11
                              12 13 14 15

            --> snake ordered:
            [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4, 5, 6, 10, 9]
        """

        ordered_lst = [i for i in range(size ** 2)]
        ordered_grid = [ordered_lst[i * size : (i + 1) * size] for i in range(size)]
        snake = []

        table = [[0] * size for _ in range(size)]
        x, y, last_dir = 0, 0, dir.UP

        def next(last_dir) -> (int, int):
            choices = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)] * 2
            for e, (x1, y1) in enumerate(choices[last_dir:]):
                if 0 <= x1 < size and 0 <= y1 < size and not table[y1][x1]:
                    return x1, y1, (last_dir + e) % 4
            else:
                return x1, y1, (last_dir + 1) % 4

        for _ in range(size ** 2):
            table[y][x] = 1
            snake.append(ordered_grid[y][x])
            x, y, last_dir = next(last_dir)

        return snake

    def neighbours(self, cost) -> list:

        neighbours = list()
        x, y = self.find(0)
        cur_pos = x + self.size * y

        def add_swapped_neigh(new_pos):
            lst = list(self.lst)
            lst[cur_pos], lst[new_pos] = lst[new_pos], lst[cur_pos]
            new_neigh = NPuzzle(size=self.size, lst=lst, local=self.local + cost)
            new_neigh.parent_sign = self.sign
            neighbours.append(new_neigh)

        if x < self.size - 1:
            add_swapped_neigh(x + 1 + self.size * y)
        if x > 0:
            add_swapped_neigh(x - 1 + self.size * y)
        if y < self.size - 1:
            add_swapped_neigh(x + self.size * (y + 1))
        if y > 0:
            add_swapped_neigh(x + self.size * (y - 1))
        return neighbours

    # Heuristic Functions

    def manhattanDistance(self, lst) -> int:
        total = 0
        for y in range(self.size):
            for x in range(self.size):
                i, j = self.find(lst[y * self.size + x])
                total += abs(j - y) + abs(i - x)
        return total

    def pythagoreanDistance(self, lst) -> float:
        total = 0
        for y in range(self.size):
            for x in range(self.size):
                i, j = self.find(lst[y * self.size + x])
                total += (abs(j - y) ** 2 + abs(i - x) ** 2) ** 0.5
        return total

    def heuristic(self) -> float:
        return self.greedy() + self.local

    # Function to modify

    def greedy(self):
        return {0: self.pythagoreanDistance, 1: self.manhattanDistance}[HEURISTIC](
            self.recorded_final_lsts[self.size]
        )


class puzzle_ordered_queue:
    """Ordered Queue made of different grid puzzles, with a dict to easily access each puzzle."""

    def __init__(self) -> None:
        self.dct = dict()
        self.visited = set()
        self.not_visited = set()
        self.lst = []

    def add(self, item: NPuzzle):

        if item.sign in self.visited:
            self.visited.remove(item.sign)

        self.dct[item.sign] = item
        self.not_visited.add(item.sign)

        def binarySearchFirstGlobLowerThan(cur_glob):
            first, last = 0, len(self.lst) - 1
            while first <= last:
                mid = (first + last) // 2
                if cur_glob <= self.lst[mid].glob:
                    last = mid - 1
                else:
                    first = mid + 1
            return last

        position = binarySearchFirstGlobLowerThan(item.glob) + 1
        self.lst.insert(position, item)

    def get_next(self):
        next = self.lst.pop(0)
        while next.sign in self.visited:
            next = self.lst.pop(0)
        self.not_visited.remove(next.sign)
        self.visited.add(next.sign)
        return next


class SolveNPuzzle:
    def __init__(self) -> None:
        self.cost = 1

        self.queue = puzzle_ordered_queue()

        self.ordered_sequence_of_states = []
        self.total_ever_selected = 0
        self.complexity_in_size = 0
        self.number_of_moves = 0

        self.npuzzle = None

    def solve(self, origin_npuzzle: NPuzzle):

        self.npuzzle = origin_npuzzle

        if not origin_npuzzle.solvable():
            return print("This NPuzzle is unsolvable")

        final_puzzle = NPuzzle(
            origin_npuzzle.size, origin_npuzzle.final_lst(), INFINITY
        )
        self.queue.add(origin_npuzzle)

        while self.queue.not_visited:

            npuzzle = self.queue.get_next()
            self.total_ever_selected += 1

            if npuzzle == final_puzzle:
                return self.success(npuzzle)
            else:
                for neigh in npuzzle.neighbours(self.cost):
                    if (
                        neigh.sign not in self.queue.dct
                        or neigh.glob < self.queue.dct[neigh.sign].glob
                    ):
                        self.queue.add(neigh)

    def success(self, final_npuzzle):

        self.complexity_in_size = len(self.queue.dct)

        self.ordered_sequence_of_states = [final_npuzzle]
        tmp = final_npuzzle
        while tmp.parent_sign:
            tmp = self.queue.dct[tmp.parent_sign]
            self.ordered_sequence_of_states.append(tmp)

        self.number_of_moves = len(self.ordered_sequence_of_states)

        return self.ordered_sequence_of_states

    def results(self):
        return f"""Solving
{str(self.npuzzle)}
total_ever_selected (time) = {self.total_ever_selected}
complexity_in_size (size) = {self.complexity_in_size}
number_of_moves = {self.number_of_moves}
"""


def get_solvable_puzzle(size) -> NPuzzle:
    my_npuzzle = NPuzzle(size)
    if not my_npuzzle.solvable():
        my_npuzzle.lst[-1], my_npuzzle.lst[0] = my_npuzzle.lst[0], my_npuzzle.lst[-1]
    return my_npuzzle


if __name__ == "__main__":

    npuzzles_to_solve = list()

    if len(sys.argv) >= 2:
        for filename in sys.argv[1:]:
            npuzzles_to_solve.append(NPuzzle(filename=filename))

    if not npuzzles_to_solve:
        npuzzles_to_solve.append(get_solvable_puzzle(DEFAULT_SIZE))

    for my_np in npuzzles_to_solve:
        solver = SolveNPuzzle()
        result = solver.solve(my_np)
        print(solver.results())
