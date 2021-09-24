#!/usr/bin/env bash

clingo regadores.lp $@ | ./parse.py > config.js