% Instructions
% ============
%
% Run this calling clingo this way,
%
% clingo \
%   -c k=10 \
%   --models 1 \
%   strips.lp \
%   strips-ayudantia.lp \
%   strips-ayudantia-0.lp

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
start(boatAt(partida)).

goal(isIn(zorro, llegada)).
goal(isIn(oveja, llegada)).
goal(isIn(trigo, llegada)).
goal(boatAt(llegada)).
