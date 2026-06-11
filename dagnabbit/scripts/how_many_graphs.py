import math


def x_choose_y(x: int, y: int) -> int:
    return math.factorial(x) / (math.factorial(y) * math.factorial(x - y))


def how_many_graphs(n: int, d: int, r: int) -> int:
    total = 0
    for i in range(n):
        choices = x_choose_y(r + i, d)
        print(f"choices: {choices}")
        total *= choices

    return total


print(how_many_graphs(16, 2, 4))
