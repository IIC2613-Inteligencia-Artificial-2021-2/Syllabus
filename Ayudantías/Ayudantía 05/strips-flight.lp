% pre-process input
could_fly(Src, Dst) :- airport(Src), airport(Dst), Src != Dst, not enemyAt(Dst), flight(Src, Dst).

% Fluents
fluent(agentAt(A)) : airport(A).

% Actions

% Definition
action(fly(Src, Dst)) :- could_fly(Src, Dst).
% Precondiciones
action_ppre(fly(Src, Dst), agentAt(Src)) :- could_fly(Src, Dst).
% Efectos
% Efectos positivos
action_add(fly(Src, Dst), agentAt(Dst)) :- could_fly(Src, Dst).
% Efectos negativos
% -2. Agente ya no esta en Src
action_del(fly(Src, Dst), agentAt(Src)) :- could_fly(Src, Dst).

