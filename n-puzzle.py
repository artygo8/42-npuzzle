#!/bin/python3
import random
from typing_extensions import final

INFINITY = 10**10


def flatten_ordered_spirale(size):

    ordered_lst = [i for i in range(size**2)]
    ordered_state = [ordered_lst[i*size:(i+1)*size] for i in range(size)]
    spirale = []

    table = [[0] * size for _ in range(size)]
    x, y, last_dir = 0, 0, dir.UP

    def next(last_dir):
        choices = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)] * 2
        for e, (x1, y1) in enumerate(choices[last_dir:]):
            if 0 <= x1 < size and 0 <= y1 < size and not table[y1][x1]:
                return x1, y1, (last_dir + e) % 4
        else:
            return x1, y1, (last_dir + 1) % 4

    for _ in range(size ** 2):
        table[y][x] = 1
        spirale.append(ordered_state[y][x])
        x, y, last_dir = next(last_dir)

    return spirale



class dir:
    UP, RIGHT, DOWN, LEFT = range(4)


class NPuzzle:

    final_lsts = dict() # size: list

    def __init__(self, size=0, lst=False, local=0):
        self.size = size or int(len(lst) ** .5)
        if self.size not in self.final_lsts:
            self.final_lsts[self.size] = self.final_lst()
        self.lst = lst or self.random_lst()
        self.sign = ' '.join(map(str, self.lst))
        self.parent_sign = None
        self.local = INFINITY if lst else 0
        self.glob = self.heuristic()

    def __eq__(self, other):
        return self.sign == other.sign

    def __str__(self):
            return '\n'.join(
                ' '.join(f'{number:{len(str(self.size**2))+1}d}' for number in self.lst[i * self.size:(i + 1) * self.size])
                for i in range(self.size)
            )

    def copy_goals(self, other):
        self.parent_sign = other.parent_sign
        self.local = other.local
        self.glob = other.glob

    def to_snake_list(self, lst=None):
        return [(lst or self.lst)[index] for index in flatten_ordered_spirale(self.size)]

    def random_lst(self):
        mess = [*range(self.size ** 2)]
        random.shuffle(mess)
        return mess

    def count_inversions(self, lst):
        tmp = list(lst)
        tmp.remove(0)
        return sum(tmp[i] > tmp[j] for i in range(len(tmp)) for j in range(i+1, len(tmp)))

    def solvable(self):
        return not bool(self.count_inversions(self.to_snake_list()) % 2)

    def final_lst(self):
        return [(j + 1, 0)[j + 1 == self.size ** 2] for j in [flatten_ordered_spirale(self.size).index(i) for i in range(self.size**2)]]

    def find(self, n: int):
        index = self.lst.index(n)
        return (index % self.size, index // self.size)

    def neighbours(self) -> set():

        neighbours = list()
        x, y = self.find(0)
        cur_pos = x + self.size * y

        def add_swapped_neigh(new_pos):
            lst = list(self.lst)
            lst[cur_pos], lst[new_pos] = lst[new_pos], lst[cur_pos]
            new_neigh = NPuzzle(size=self.size, lst=lst, local=self.local + 1)
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

    def manhattanDistance(self, lst) -> int:
        total = 0
        for y in range(self.size):
            for x in range(self.size):
                i, j = self.find(lst[y * self.size + x])
                total += (abs(j - y) + abs(i - x))
        return total

    def pythagoreanDistance(self, lst):
        total = 0
        for y in range(self.size):
            for x in range(self.size):
                i, j = self.find(lst[y * self.size + x])
                total += (abs(j - y)**2 + abs(i - x)**2)**.5
        return total

    def heuristic(self):
        return self.pythagoreanDistance(self.final_lsts[self.size]) + ((self.local or 1) ** .5)
        


class puzzle_ordered_queue:
    """Ordered Queue made of different state puzzles, with a dict to easily access each puzzle."""

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
        self.nodes = dict()  # key=NPuzzle.sign, value=NPuzzle
        self.cost = 0
        self.max_size = 0
        # self.complexity_in_size = 0
        self.queue = puzzle_ordered_queue()


    def solve(self, origin_npuzzle: NPuzzle):

        if not origin_npuzzle.solvable():
            return print('This NPuzzle is impossible')

        final_puzzle = NPuzzle(origin_npuzzle.size, origin_npuzzle.final_lst(), INFINITY)
        self.queue.add(origin_npuzzle)

        while self.queue.not_visited:
            npuzzle = self.queue.get_next()
            if npuzzle == final_puzzle:
                return len(self.queue.dct)
            else:
                for neigh in npuzzle.neighbours():
                    if neigh.sign not in self.queue.dct or neigh.glob < self.queue.dct[neigh.sign].glob:
                        self.queue.add(neigh)


def get_solvable_puzzle(size):
    my_npuzzle = NPuzzle(size)
    if not my_npuzzle.solvable():
        my_npuzzle.lst[-1], my_npuzzle.lst[0] = my_npuzzle.lst[0], my_npuzzle.lst[-1]
    return my_npuzzle


if __name__ == '__main__':
    results = []
    size = 3
    for _ in range(100):
        my_npuzzle = get_solvable_puzzle(size)
        solver = SolveNPuzzle()
        results.append(solver.solve(my_npuzzle))
        # print(results)
    print(results)
    print(f'complexity in size: mean={sum(results) // len(results)} median={sorted(results)[len(results)//2]}')
