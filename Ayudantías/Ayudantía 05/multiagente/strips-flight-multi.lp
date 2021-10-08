% pre-process input
could_fly(Src, Dst) :- airport(Src), airport(Dst), Src != Dst, flight(Src, Dst).
flight(Dst, Src) :- flight(Src, Dst).

% Fluents
fluent(agentAt(Ag, A)) : airport(A), agent(Ag).

% Actions

% Definition
action(fly(Src, Dst)) :- could_fly(Src, Dst).
% Precondiciones
action_ppre(Ag, fly(Src, Dst), agentAt(Ag, Src)) :- could_fly(Src, Dst), agent(Ag).
% Efectos
% Efectos positivos
action_add(Ag, fly(Src, Dst), agentAt(Ag, Dst)) :- could_fly(Src, Dst), agent(Ag).
% Efectos negativos
% -2. Agente ya no esta en Src
action_del(Ag, fly(Src, Dst), agentAt(Ag, Src)) :- could_fly(Src, Dst), agent(Ag).

:- holds(T, agentAt(Ag1, A)), holds(T, agentAt(Ag2, A)), Ag1 != Ag2, agent(Ag1), agent(Ag2), airport(A), time(T).