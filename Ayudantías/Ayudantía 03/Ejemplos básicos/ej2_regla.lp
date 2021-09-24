% Declaramos que tom es un gato
gato(tom).

% Para definir una regla usamos la siguiente sintaxis.
% En este caso estoamos diciendo si X es gato entonces X es animal.
animal(X) :- gato(X).