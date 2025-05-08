import numpy as np
from player import Player

class Kondys_Dabrowski(Player):
    """Gracz w oszusta, którego strategia opiera się o liczbę kart w ręce oraz o figurę karty przeciwnika. 
    Macierz wypłat zmienia się w trakcie zadania przez, co dostosowywana jest taktyka pod nieznaną macierz wypłat.
    Gracz operuje na prawdopodobieństwach odnoszących się do wykonywania odpowiednich akcji. Strategią dominującą
    jest zagranie najniższej karty w momencie gdy możemy ją zagrać. Jeśli nasza najmniejsza karta ma figurę niższą
    od figury na stole gracz ma możliwość oszukania. Oszukanie w tym momencie to deklaracja najbliższej karty z ręki
    i postawienie karty najmniejszej. Gracz ma też możliwość nie oszukania i wystawienia karty najbliższej tej na stole.
    Jeśli nie ma karty do oszukania to deklaruje losową wyższą lub równą kartę zadeklarowanej.
    Dodatkowo zaimplementowano strategię, która dla dużej liczby kart w ręce sprawia, że gracz wybiera wyższe karty, aby
    potem zmuszać przeciwnika do ciągnięcia kart, lub oszukiwania.
    """
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.cards: list[tuple[int, int]] = [] # edycja konstruktora klasy w celu pozbycia się komunikatów w IDE.
        self.seen: list[tuple[int, int]] = [] # dodanie tablicy z widzianymi kartami
    def lowestCardDeclaired(self, declared_card: tuple[int, int]) -> tuple[int, int]:
        
        """Funkcja do wybierania karty o wartości najbliższej karcie zadeklarowanej.

        Args:
            declared_card (tuple[int, int]): Zadeklarowana karta.
        
        Returns:
            tuple[int, int]: Najbliższa karta z ręki do zadeklarowanej.
        """
        def sorter(x: tuple[int, int]) -> int | float:
            """Metoda pomocnicza do wybierania karty o wartości najbliższej karcie zadeklarowanej.

            Args:
                x (tuple[int, int]): Karta do porównania 

            Returns:
                int|float: Wartość figury większej lub równej karcie zadeklarowanej 
            """
            if x[0] >= declared_card[0]:
                return x[0]
            return np.inf
        return min([(-1,0)] + self.cards, key= lambda x: sorter(x))
    def putCard(self, declared_card: tuple[int,int]|None) -> tuple[tuple[int,int], tuple[int, int]]|str:
        """Określa decyzję gracza o wystawieniu karty poprawnej, oszukaniu lub ciągnięciu kart. Decyzja jest oparta na
        liczebności kart w ręce gracza. Strategia przyjmuje większe prawdopodobieństwo oszustwa dla małej liczby kart.
        Dodatkowo jeśli nasz gracz ma dużo kart to przyjmuje strategię wystawiania dużych wartości figur do zmuszania
        przeciwnika do ciągnięcia kart lub oszustwa. Dobieranie kart rozpatrujemy tylko wtedy kiedy zostaje nam jedna karta.
        Dobieranie jest strategią ściśle zdominowaną, ponieważ jak oszukamy to w najgorszym wypadku skończymy z jedną kartą,
        mniej niż przy dobieraniu.

        Args:
            declared_card (tuple[int,int] | None): Karta na stole.

        Returns:
            tuple[tuple[int,int], tuple[int, int]]|str: Deklaracja karty lub decyzja o ciągnięciu kart.
        """

        prob: float
        idx: int
        choiche: tuple[tuple[int,int], tuple[int, int]]
        minimal_card: tuple[int, int]
        fake_card: tuple[int, int]
        num_of_cards: int = len(self.cards)

        self.seen.extend(self.cards)
        self.seen = list(set(self.seen)) # kompresja listy widzianych kart

        # strategia zakładająca wystawianie wysokich kart w momencie, gdy mamy dużo kart
        if num_of_cards > 10:
            if declared_card is not None and len(np.array(self.cards)[np.array(self.cards)[:,0] >= declared_card[0]]) > 8:
                return sorted(self.cards, key=lambda x: x[0])[::-1][4], sorted(self.cards, key=lambda x: x[0])[::-1][4]
        
        minimal_card = min(self.cards, key=lambda x: x[0]) # znajdywanie najmniejszej karty

        # dostosowanie prawdopodobieństwa do liczby kart w ręce
        if declared_card is None or minimal_card[0] >= declared_card[0]:
            return minimal_card, minimal_card
        if num_of_cards < 2: # dla małej liczby kart taktyka bardziej ryzykowna
            prob = 0.8
        elif num_of_cards < 4: 
            prob = 0.6
        elif num_of_cards < 7:
            prob = 0.3
        elif num_of_cards < 10: # wraz ze wzrostem liczby kart zmniejszamy szansę na ruchy ryzykowne
            prob = 0.1
        else:
            prob = 0
        
        # wybór karty do potencjalnego oszukania
        fake_card = self.lowestCardDeclaired(declared_card)
        if fake_card[0] == -1: # jeśli nie jesteśmy w stanie wybrać karty do oszustwa, wybieramy losowo
            if num_of_cards != 1: # nie możemy oszukać ostatniej karty
               # dobieranie kart jest strategią ściśle zdominowaną
               random_card = np.random.randint(declared_card[0], 15), np.random.randint(0,4)
               i = 0
               while random_card in self.seen: # jeśli losowa karta była widziana powtarza losowanie
                   random_card = np.random.randint(declared_card[0], 15), np.random.randint(0,4)
                   if i == 1000: # wszystkie karty widziane, brak możliwości bezkarnego oszukania
                       break
                   i += 1
               return minimal_card, random_card
            return 'draw'
        
        idx = np.random.choice([0,1], p = [prob, 1-prob])
        choiche = [(minimal_card, fake_card), (fake_card, fake_card)][idx] # decyja czy wystawiamy kartę, czy oszukujemy
        return choiche[0], choiche[1]
    
    def checkCard(self, opponent_declaration: tuple[int, int]) -> bool:
        """Metoda do sprawdzania, czy przeciwnik oszukał czy nie. Decyzja jest podejmowana na podstawie figury przeciwnika.
        Funkcja zakłada, że przeciwnik jest bardziej skłonny do oszustwa przy deklaracji wysokiej karty. Stawianie wartości
        niskiej jest mało prawdopodobne do oszukania. Dodatkowo jest zabezpieczenie przed deklaracją karty zawartej w ręce.

        Args:
            opponent_declaration (tuple[int, int]): _description_

        Returns:
            bool: Decyzja o sprawdzeniu czy przeciwnik oszukał (True - oszukał, False - nie oszukał).
        """
        
        prob: float = 0
        if opponent_declaration in self.cards: 
            return True # sprawdzenie czy zadeklarowana karta przeciwnika jest w twojej ręce
        match opponent_declaration[0]: # wybór prawdopodobieństwa na bazie figury przeciwnika
            case 9:
                prob = 0
            case 10:
                prob = 0
            case 11:
                prob = 0.2
            case 12:
                prob = 0.5
            case 13:
                prob = 0.8
            case 14:
                prob = 0.9
        
        return np.random.choice([True, False], p=[prob, 1-prob])
