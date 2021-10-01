% Instructions
% ============
%
% Run this calling clingo this way,
%
% clingo \
%   -c k=10 \
%   --models 1 \
%   strips.lp \
%   strips-acertijo.lp \
%   strips-acertijo-0.lp

boat(b1).
boat(b2).

ser_vivo(zorro).
ser_vivo(oveja).
ser_vivo(trigo).

costa(partida).
costa(llegada).

conflict(zorro, oveja).
conflict(oveja, trigo).

start(isIn(zorro, partida)).
start(isIn(oveja, partida)).
start(isIn(trigo, partida)).
start(boatAt(b1, partida)).
start(boatAt(b2, partida)).

goal(isIn(zorro, llegada)).
goal(isIn(oveja, llegada)).
goal(isIn(trigo, llegada)).
start(boatAt(b1, llegada)).
start(boatAt(b2, llegada)).
