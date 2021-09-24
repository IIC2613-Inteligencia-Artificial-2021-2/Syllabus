% Declaramos dos predicados
p(2).
q(1).

% Para hacer el or entre los predicados podemos usar dos reglas distintas (o más para más predicados)
% donde cada regla utiliza uno de los predicados.
or(X) :- p(X).
or(X) :- q(X).

% Con esta linea decimos que queremos que en el output del programa
% se impriman solo los predicados "or" el "/1" indica la cantidad de argumentos del predicado.
#show or/1.