% Si S es un regador entonces tiene que estar en alguna posicion dentro del tablero.
{ on(S, X, Y) : rangeX(X), rangeY(Y) } = 1 :- sprinkler(S).

% No pueden haber dos regadores en la misma posicion.
:- on(S1, X, Y), on(S2, X, Y), S1 != S2.

% Un regador no puede estar encima de un obstaculo
:- on(S,X,Y), obstacle(X, Y).

% Definimos las celdas que son regables como las que están dentro del mapa y no están ocupadas
irrigable(X, Y) :- rangeX(X), rangeY(Y), sprinkler(S), not on(S, X, Y), not obstacle(X, Y).

% Definimos las celdas que son regadas como las celdas que estan alrededor de un regador
irrigated(X, Y) :- on(S, X - 1, Y    ), sprinkler(S), irrigable(X, Y).
irrigated(X, Y) :- on(S, X + 1, Y    ), sprinkler(S), irrigable(X, Y).
irrigated(X, Y) :- on(S, X    , Y - 1), sprinkler(S), irrigable(X, Y).
irrigated(X, Y) :- on(S, X    , Y + 1), sprinkler(S), irrigable(X, Y).
irrigated(X, Y) :- on(S, X - 1, Y - 1), sprinkler(S), irrigable(X, Y).
irrigated(X, Y) :- on(S, X - 1, Y + 1), sprinkler(S), irrigable(X, Y).
irrigated(X, Y) :- on(S, X + 1, Y - 1), sprinkler(S), irrigable(X, Y).
irrigated(X, Y) :- on(S, X + 1, Y + 1), sprinkler(S), irrigable(X, Y).

% Descartamos los modelos en donde caiga agua fuera del mapa para evitar mojar gente :D
:- sprinkler(S), on(S, X, Y), not rangeX(X - 1).
:- sprinkler(S), on(S, X, Y), not rangeX(X + 1).
:- sprinkler(S), on(S, X, Y), not rangeY(Y - 1).
:- sprinkler(S), on(S, X, Y), not rangeY(Y + 1).

% Optimizamos para que haya la mayor cantidad de celdas regadas (afecta rendimiento)
#maximize{1, X, Y: irrigated(X, Y)}.

#show sprinkler/1.
#show rangeX/1.
#show rangeY/1.
#show obstacle/2.
#show on/3.
