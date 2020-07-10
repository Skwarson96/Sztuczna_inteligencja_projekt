# Lokalizacja

Model przejścia:
Macierz o wymiarach (4, 42, 42)
Początkowo są 4*42 stany robota, dlatego na początku wszystkie stany mają takie same prawdopodobieństwo
1/168.
W momencie gdy robot wykonuje krok w przód prawdopodobieństwo że się przesunie wynosi 0.95, natomiast że zostanie w miejscu
0.05. Gdy robot wykonuje obrót macierz przejścia jest macierzą jednostkową, ponieważ robot nie zmienia pozycji.

Model sensora:
Macierz o wymiarch (4, 42, 1)
Dla każdego możliwego kierunku każdego możliwego pola obliczane jest prawdopodobieństwo na podstawie aktualnego stanu 
wykrytego przez czujniki. Gdy stan z percept zgadza się z rzeczywistością, to znaczy wykryta przeszkoda faktycznie, na podstawie mapy,
znajduje się przed robotem to prawdopodobieństwo dla tego pola wynosi 0.9. Gdy stan wykryty przez czujnik nie pokrywa się z mapą
to prawdopodobieństwo jest równe 0.1.
Gdy czujnik nie wykryje żadnej przeszkody to przyjmowana jest macierz z poprzedniego stanu.


# Heurystyka

Heurystyka działa na kilku zasadach:

1 - gdy robot wykryje, że jest w ślepej uliczce ( percept == ['fwd', 'right', 'left'] ) wykona zwrot, który składa
się z dwóch obrotów w lewo a następnie zrobi krok do przodu. Dzięki temu sprawnie wychodzi ze ślepych zaułków.

2 - gdy robot wykryje przeszkodę przed sobą wykona zwrot w lewo lub prawo z prawdopodobieństwem 50% dla każdej z opcji.
Jednak robot nie wykona obrotu w lewo jeżeli poprzednią akcją był obrót w prawo i analogicznie nie obróci się w prawo jeżelii
wcześniej obrócił się w lewo. Dzięki temu robot nie wykonuje bezsensownych obrotów przed przeszkodą, tylko zawróci.

3 - gdy robot wykryje przeszkodę po prawej stronie lub z przodu i po prawej stronie ( analogicznie dla lewej strony) 
to wykona obrót w lewo lub krok do przodu z prawdopodobieństwem 50% dla każdej z opcji.

4 - gdy nie zostanie spełniony żadny z powyższych warunków np.: czujnik nie wykryje żadnej przeszkody to robot wykona krok do przodu.
