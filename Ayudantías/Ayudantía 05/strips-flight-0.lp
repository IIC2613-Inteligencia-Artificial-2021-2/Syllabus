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

airport(santiago).
airport(madrid).
airport(miami).
airport(rio).
airport(paris).
airport(londres).
airport(roma).
airport(venecia).

enemyAt(paris).
enemyAt(londres).

flight(santiago, rio).
flight(rio, roma).
flight(roma, venecia).
flight(venecia, madrid).
flight(rio, miami).
flight(miami, londres).
flight(santiago, paris).
flight(santiago, miami).
flight(paris, roma).

start(agentAt(santiago)).

goal(agentAt(madrid)).
