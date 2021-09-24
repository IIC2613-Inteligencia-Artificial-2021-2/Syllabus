% Se define un mapa cuadrado como una matriz de m x m, donde el valor de una casilla está dado por 
% el par (X, Y). En este caso se tiene un mapa de 2x2, donde el rango de X e Y va de 0 a 1:
rangeX(0..1).
rangeY(0..1).

% En este mapa existe una trampa en la casilla (0, 0) y se tienen a disposición 3 soldados:
trap(0, 0).
soldier(1..3).

% Para cada uno de los soldados se debe escoger una única posición dentro del mapa:
1 {on(X, Y, S) : rangeX(X), rangeY(Y)} 1 :- soldier(S).

% Se descartan los modelos donde existan dos soldados que hayan sido ubicados en la misma casilla o 
% en donde exista una trampa:
:- on(X, Y, S1), on(X, Y, S2), S1 != S2.
:- on(X, Y, S), trap(X, Y).

% Nos interesa solo ver las posiciones donde se ubican los soldados en los distintos modelos:
#show on/3.
