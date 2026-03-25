% ============================================================
% ARITHMETIC OPERATIONS
% ============================================================

add(X, Y, R) :- R is X + Y.

subtract(X, Y, R) :- R is X - Y.

multiply(X, Y, R) :- R is X * Y.

divide(X, Y, R) :-
    Y \= 0,
    R is X / Y.

modulo(X, Y, R) :-
    Y \= 0,
    R is X mod Y.

power(X, Y, R) :- R is X ** Y.

% ============================================================
% SET OPERATIONS
% ============================================================

% Union: all elements from both lists, no duplicates
union([], L, L).
union([H|T], L, R) :-
    member(H, L),
    union(T, L, R).
union([H|T], L, [H|R]) :-
    \+ member(H, L),
    union(T, L, R).

% Intersection: only elements present in both lists
intersection([], _, []).
intersection([H|T], L, [H|R]) :-
    member(H, L),
    intersection(T, L, R).
intersection([H|T], L, R) :-
    \+ member(H, L),
    intersection(T, L, R).

% Difference: elements in L1 but not in L2
difference([], _, []).
difference([H|T], L, R) :-
    member(H, L),
    difference(T, L, R).
difference([H|T], L, [H|R]) :-
    \+ member(H, L),
    difference(T, L, R).

% Subset: true if every element of L1 exists in L2
subset([], _).
subset([H|T], L) :-
    member(H, L),
    subset(T, L).

% Symmetric Difference: elements in L1 or L2 but not both
symmetric_diff(L1, L2, R) :-
    difference(L1, L2, D1),
    difference(L2, L1, D2),
    append(D1, D2, R).

% Disjoint: true if L1 and L2 share no common elements
disjoint([], _).
disjoint([H|T], L) :-
    \+ member(H, L),
    disjoint(T, L).

% ============================================================
% SAMPLE QUERIES
% ============================================================
% ?- add(7, 5, R).            % R = 12
% ?- subtract(10, 4, R).      % R = 6
% ?- multiply(6, 9, R).       % R = 54
% ?- divide(20, 4, R).        % R = 5.0
% ?- modulo(17, 5, R).        % R = 2
% ?- power(2, 8, R).          % R = 256.0
%
% ?- union([1,2,3],[3,4,5],R).           % R = [1,2,3,4,5]
% ?- intersection([1,2,3],[3,4,5],R).    % R = [3]
% ?- difference([1,2,3],[3,4,5],R).      % R = [1,2]
% ?- subset([2,3],[1,2,3,4]).            % true
% ?- symmetric_diff([1,2,3],[3,4,5],R). % R = [1,2,4,5]
% ?- disjoint([1,2,3],[4,5,6]).          % true
