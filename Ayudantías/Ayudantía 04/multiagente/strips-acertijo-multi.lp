% pre-process input
could_carry(B, Src, Dst, SerVivo) :- could_sail(B, Src, Dst), ser_vivo(SerVivo), boat(B).
could_carry(B, Src, Dst, SerVivo) :- could_carry(B, Dst, Src, SerVivo), boat(B).
could_sail(B, Src, Dst) :- costa(Src), costa(Dst), Src != Dst, boat(B).
conflict(V1, V2) :- conflict(V2, V1).
agent(Agent) :- boat(Agent).
boat(Agent) :- agent(Agent).

% Fluents
fluent(isIn(S, C)) : ser_vivo(S), costa(C).
fluent(boatAt(B, C)) : costa(C), boat(B).


% Carry some living being through the river
% ----
% Definition
action(carry(B, Src, Dst, SerVivo)) :- could_carry(B, Src, Dst, SerVivo).
% Precondiciones
% +1. El bote se encuentra en Src
action_ppre(B, carry(B, Src, Dst, SerVivo), boatAt(B, Src)) :- could_carry(B, Src, Dst, SerVivo).
% +2. El ser vivo se encuentra en Src
action_ppre(B, carry(B, Src, Dst, SerVivo), isIn(SerVivo, Src)) :- could_carry(B, Src, Dst, SerVivo).
% Efectos
% Efectos positivos
% +1. Bote queda en Dst
action_add(B, carry(B, Src, Dst, SerVivo), boatAt(B, Dst)) :- could_carry(B, Src, Dst, SerVivo).
% +2. Ser vivo queda en Dst
action_add(B, carry(B, Src, Dst, SerVivo), isIn(SerVivo, Dst)) :- could_carry(B, Src, Dst, SerVivo).
% Efectos negativos
% -1. Bote ya no está en Src
action_del(B, carry(B, Src, Dst, SerVivo), boatAt(B, Src)) :- could_carry(B, Src, Dst, SerVivo).
% -2. Ser vivo ya no está en Src
action_del(B, carry(B, Src, Dst, SerVivo), isIn(SerVivo, Src)) :- could_carry(B, Src, Dst, SerVivo).

% Sail through the river carring no living being.
% ----
% Definition
action(sail(B, Src, Dst)) :- could_sail(B, Src, Dst).
% Precondiciones
% +1. El bote se encuentra en Src
action_ppre(B, sail(B, Src, Dst), boatAt(B, Src)) :- could_sail(B, Src, Dst).
% Efectos
% Efectos positivos
% +1. Bote queda en Dst
action_add(B, sail(B, Src, Dst), boatAt(B, Dst)) :- could_sail(B, Src, Dst).
% Efectos negativos
% -1. Bote ya no está en Src
action_del(B, sail(B, Src, Dst), boatAt(B, Src)) :- could_sail(B, Src, Dst).

% Other restriccions
:- holds(T, isIn(V1, C)), holds(T, isIn(V2, C)), not holds(T, boatAt(B, C)), conflict(V1,V2), time(T), boat(B).
