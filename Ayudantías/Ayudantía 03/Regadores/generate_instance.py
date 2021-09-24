import random

sprinklers = 4
width = 10
height = 10
proportion = 0.15
target_file = "instance.lp"

samples = int(proportion * width*height)

cells = [(row, col) for row in range(width) for col in range(height)]
obstacles = random.choices(cells, k=samples)


with open(target_file, "w") as f:
    f.write(f"sprinkler(1..{sprinklers}).")
    f.write(f"rangeX(0..{width-1}).\n")
    f.write(f"rangeY(0..{height-1}).\n\n")
    for row, col in obstacles:
        f.write(f"obstacle({row},{col}).\n")
