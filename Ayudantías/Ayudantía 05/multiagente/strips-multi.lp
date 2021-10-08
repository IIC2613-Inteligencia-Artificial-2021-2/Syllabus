% Time
% ====
time(0..k-1).

% Actions
% =======
0 { exec(T, Agent, Action) : action(Action) } 1 :- time(T), agent(Agent).


% State evolution
% ===============

% Actions chosen respect the prerequisites
% Positive prerequisites
:- exec(T, Agent, Action), action_ppre(Agent, Action, Fluent), not holds(T, Fluent) : agent(Agent).
% Negative prerequisites
:- exec(T, Agent, Action), action_npre(Agent, Action, Fluent), holds(T, Fluent) : agent(Agent).

% State evolution
% A fluent holds if it was added by the last action.
holds(T+1, Fluent) :- exec(T, Agent, Action),
                      action_add(Agent, Action, Fluent).
% Or if it was true and it was not made false by the  last action.
fluent_dropped(T, Fluent) :-
  time(T),
  exec(T, Agent, Action),
  action_del(Agent, Action, Fluent).

holds(T+1, Fluent) :-
  time(T),
  agent(Agent),
  holds(T, Fluent),
  not fluent_dropped(T, Fluent).
  % A fluent holds if no action has been executed
some_action_executed(T) :- exec(T, Agent, Action).
holds(T+1, Fluent) :- holds(T, Fluent),
                      time(T),
                      not some_action_executed(T).


% Problem definition
% ==================

% Initial state
holds(0, Fluent) :- start(Fluent).
% Goal condition
:- goal(Fluent), not holds(k, Fluent).

% Describe only the world and the solution
%#show holds/2.
#show exec/3.

% Problem integrity checks
% ========================

% Problems must be stated correctly.
% 1. An action can't both, add and delete the same Fluent.
:- action_add(Agent1, Action, Fluent), action_del(Agent2, Action, Fluent),
   agent(Agent1),
   agent(Agent2).
% 2. A fluent cannot be both, a positive and a negative precondition.
:- action_ppre(Agent1, Action, Fluent), action_npre(Agent2, Action, Fluent),
   agent(Agent1),
   agent(Agent2).
% 3. A fluent cannot be added by two agents at the same time
:- time(T),
 agent(Agent1),
 agent(Agent2),
 Agent1!=Agent2,
 exec(T, Agent1, Action1),
 exec(T, Agent2, Action2),
 action_add(Agent1, Action1, Fluent),
 action_add(Agent2, Action2, Fluent).
% 4. A fluent cannot be made false by two agents at the same time
:- time(T),
   agent(Agent1),
   agent(Agent2),
   Agent1!=Agent2,
   exec(T, Agent1, Action1),
   exec(T, Agent2, Action2),
   action_del(Agent1, Action1, Fluent),
   action_del(Agent2, Action2, Fluent).
