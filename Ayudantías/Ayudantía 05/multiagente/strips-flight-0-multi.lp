% Instructions
% ============
%
% Run this calling clingo this way,
%
% clingo \
%   -c k=10 \
%   --models 1 \
%   strips.lp \
%   strips-ayudantia.lp \
%   strips-ayudantia-0.lp

agent(pedro).
agent(juan).
agent(diego).

airport(santiago).
airport(madrid).
airport(miami).
airport(rio).
airport(paris).
airport(londres).
airport(roma).
airport(venecia).


flight(santiago, rio).
flight(rio, roma).
flight(roma, venecia).
flight(venecia, madrid).
flight(rio, miami).
flight(miami, londres).
flight(santiago, paris).
flight(santiago, miami).
flight(paris, roma).

start(agentAt(pedro, santiago)).
start(agentAt(juan, miami)).
start(agentAt(diego, paris)).


goal(agentAt(pedro, madrid)).
goal(agentAt(juan, santiago)).
goal(agentAt(diego, roma)).