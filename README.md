# lokalizacja

Tutaj umieść swój raport.

# heurystyka

Heurystyka działa na kilku zasadach:

1 - gdy robot wykryje, że jest w ślepej uliczce ( percept == ['fwd', 'right', 'left']) wykona zwrot, który składa
się z dwóch obrotów w lewo a następnie zrobi krok do przodu. Dzięki temu sprawnie wychodzi ze ślepych zaułków.

2 - gdy robot wykryje przeszkodę przed sobą wykona zwrot w lewo lub prawo z prawdopodobieństwem 50% dla każdej z opcji.
Jednak robot nie wykona obrotu w lewo jeżeli poprzednią akcją był obrót w prawo i analogicznie nie obróci się w prawo jeżelii
wcześniej obrócił się w lewo. Dzięki temu robot nie wykonuje bezsensownych ruchów.

3 - gdy robot wykryje przeszkodę po prawej stronie lub z przodu i po prawej stronie ( analogicznie dla lewej strony) 
to wykona obrót w lewo lub krok do przodu z prawdopodobieństwem 50% dla każdej z opcji.

4 - gdy nie zostanie spełniony żadny z powyższych warunków robot wykona krok do przodu.
