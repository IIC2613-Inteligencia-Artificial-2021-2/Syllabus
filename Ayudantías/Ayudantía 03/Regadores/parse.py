#!/usr/bin/env python3
import sys
import re


def scan_numbers(str):
    numbers = re.findall('[0-9]+', str)
    return [int(n) for n in numbers]


lines = sys.stdin.readlines()
idx = 4
for i, line in enumerate(lines):
    if "Answer" in line:
        idx = i + 1
results = lines[idx].strip().split(" ")

rangeX_list = [elem for elem in results if "rangeX" in elem]
rangeY_list = [elem for elem in results if "rangeY" in elem]
sprinkler_list = [elem for elem in results if "sprinkler" in elem]
obstacle_list = [elem for elem in results if "obstacle" in elem]
on_list = [elem for elem in results if "on" in elem]

rangeX = max(map(lambda x: scan_numbers(x)[0], rangeX_list))
rangeY = max(map(lambda x: scan_numbers(x)[0], rangeY_list))
sprinklers = max(map(lambda x: scan_numbers(x)[0], sprinkler_list))
obstacles = list(map(scan_numbers, obstacle_list))
on = list(map(scan_numbers, on_list))

print(f"const rangeX = {rangeX}")
print(f"const rangeY = {rangeY}")
print(f"const sprinklers = {sprinklers}")
print(f"const obstacles = {obstacles}")
print(f"const on = {on}")
