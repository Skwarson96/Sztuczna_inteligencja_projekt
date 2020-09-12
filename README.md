# Lokalizacja

Model przejścia:
W momencie gdy robot wykonuje krok w przód prawdopodobieństwo że się przesunie wynosi 0.95, natomiast że zostanie w miejscu
0.05. Gdy robot wykonuje obrót macierz przejścia jest macierzą jednostkową, ponieważ robot nie zmienia pozycji. Jednak w tym przypadku konieczna jest zmiana w macierzy self.P. Zmieniana jest kolejność kolumn w odpowienią stronę w zależności od tego czy został wykonany obrót w lewo czy w prawo. 

Bump:
Gdy zostanie wykryte 'bump' robot zostaje w tym samym miejscu, macierz tranzycji jest wtedy macierzą jednostkową.

Model sensora:
Dla każdego możliwego kierunku każdego możliwego pola obliczane jest prawdopodobieństwo na podstawie aktualnego stanu 
wykrytego przez czujniki. Gdy stan z percept zgadza się z rzeczywistością, to znaczy, że wykryta przeszkoda faktycznie, na podstawie mapy,
znajduje się przed robotem. Wtedy prawdopodobieństwo dla tego pola wynosi 0.9. Gdy stan wykryty przez czujnik nie pokrywa się z mapą
to prawdopodobieństwo jest równe 0.1.
Gdy czujnik nie wykryje żadnej przeszkody to przyjmowana jest macierz z poprzedniego stanu.


# Heurystyka

Heurystyka prowadzi robota w taki sposób aby trzymał się przy ścianie. Gdy robot będzie miał ponad 80% pewności, że znajduje się w danym miejscu zaczyna się poruszać w sposób bardziej losowy.
