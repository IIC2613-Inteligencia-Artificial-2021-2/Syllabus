p(1..2).
q(1).

% Para hacer el and entre dos predicados lo hacemos utilizando una coma entre ellos.
and(X) :- p(X), q(X).

#show and/1.