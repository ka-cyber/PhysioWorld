# Policy guide

Fixed, rule, greedy, MPC, bandit, Q-learning, SARSA, and finite dynamic
programming policies share an action interface. The finite DP implementation
is an oracle only for its explicitly declared finite MDP. Other policies must
be compared against the same objective and constraints; they are not called
optimal by convention.

Capability support uses independent optimization for each representation.
The same policy ID is never implicitly reused for a support-function gap.
