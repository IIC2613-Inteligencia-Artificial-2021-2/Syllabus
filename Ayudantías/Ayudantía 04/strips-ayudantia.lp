% pre-process input
could_carry(Src, Dst, S) :- costa(Src), costa(Dst), ser_vivo(S), Src != Dst.
could_sail(Src, Dst) :- costa(Src), costa(Dst), Src != Dst.
conflict(V1, V2) :- conflict(V2, V1).

% Fluents
fluent(isIn(S, C)) : ser_vivo(S), costa(C).
fluent(boatAt(C)) : costa(C).

% Actions

% Definition
action(carry(Src, Dst, S)) :- could_carry(Src, Dst, S).
% Precondiciones
action_ppre(carry(Src, Dst, S), boatAt(Src)) :- could_carry(Src, Dst, S).
action_ppre(carry(Src, Dst, S), isIn(S, Src)) :- could_carry(Src, Dst, S).
% Efectos
% Efectos positivos
action_add(carry(Src, Dst, S), boatAt(Dst)) :- could_carry(Src, Dst, S).
action_add(carry(Src, Dst, S), isIn(S, Dst)) :- could_carry(Src, Dst, S).
% Efectos negativos
% -1. Bote ya no está en Src
action_del(carry(Src, Dst, S), boatAt(Src)) :- could_carry(Src, Dst, S).
% -2. Ser vivo ya no está en Src
action_del(carry(Src, Dst, S), isIn(S, Src)) :- could_carry(Src, Dst, S).

% Sail through the river carring no living being.
% ----
% Definition
action(sail(Src, Dst)) :- could_sail(Src, Dst).
% Precondiciones
% +1. El bote se encuentra en Src
action_ppre(sail(Src, Dst), boatAt(Src)) :- could_sail(Src, Dst).
% Efectos
% Efectos positivos
% +1. Bote queda en Dst
action_add(sail(Src, Dst), boatAt(Dst)) :- could_sail(Src, Dst).
% Efectos negativos
% -1. Bote ya no está en Src
action_del(sail(Src, Dst), boatAt(Src)) :- could_sail(Src, Dst).


% Other restriccions
:- holds(T, isIn(V1, C)), holds(T, isIn(V2, C)), not holds(T, boatAt(C)), conflict(V1,V2), time(T).
