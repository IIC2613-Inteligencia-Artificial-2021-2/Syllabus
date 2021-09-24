% Se puede definir una serie de, por ejemplo, direcciones como arriba, abajo, etc.
direction(up).
direction(down).
direction(left).
direction(right).

% Luego, se puede forzar con una restricción de cardinalidad para que el programa escoja una única 
% dirección para ejecutar un movimiento:
1 {move(A) : direction(A)} 1.

% Podemos restringir el output del modelo para que solo muestre la variable de interés:
#show move/1.
