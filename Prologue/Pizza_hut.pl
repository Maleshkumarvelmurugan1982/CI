:- dynamic item/2.

% add a new item
add_item(Name, Price) :-
    assertz(item(Name, Price)).

% remove an item
remove_item(Name) :-
    retract(item(Name, _)).

% update item price
update_item(Name, NewPrice) :-
    retract(item(Name, _)),
    assertz(item(Name, NewPrice)).

% show all menu items
show_menu :-
    forall(item(Name, Price),
        (write(Name), write(' - Rs '), write(Price), nl)).

% process order list recursively
order_list([], 0).
order_list([(Item, Qty)|T], Total) :-
    item(Item, Price),
    Sub is Price * Qty,
    write('item:     '), write(Item), nl,
    write('qty:      '), write(Qty), nl,
    write('subtotal: Rs '), write(Sub), nl, nl,
    order_list(T, Rest),
    Total is Sub + Rest.

% generate full bill
bill(OrderList) :-
    order_list(OrderList, Total),
    write('--- total bill: Rs '), write(Total), write(' ---'), nl.

% sample usage (after adding items dynamically):
% ?- add_item(dosa, 80).
% ?- add_item(idli, 50).
% ?- add_item(coffee, 40).
% ?- bill([(dosa,2),(idli,3),(coffee,1)]).
