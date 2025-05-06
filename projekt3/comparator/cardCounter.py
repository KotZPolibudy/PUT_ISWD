import numpy as np
from player import Player


class FirstCardCounter(Player):
    def __init__(self, name):

        super().__init__(name)
        # startowe wartości
        self.shouldCheat = 0.01
        self.checkProb = 0.2
        self.cheat = False
        self.randomlyPlayTopCard = 0.02
        self.learningRate = 0.3
        self.roundsCounter = 0
        self.card_pile = []
        self.last_opp_card = None
        self.opponent_declarations = []

    def putCard(self, declared_card):
        # never return "draw" :))
        self.cards.sort(key=lambda x: x[0])
        self.cheat = False

        # Jeśli pierwszy ruch, to nie ma co oszukiwać, graj najmniejszą kartę.
        if declared_card is None:
            return self.cards[0], (self.cards[0][0], self.cards[0][1])

        # NEVER DRAW — ale gra mówi, że ostatnia karta MUSI być fair, huh?
        # to jest na ostatnią kartę
        if len(self.cards) == 1:
            if self.cards[0][0] < declared_card[0]:
                return "draw"
            else:
                return self.cards[0], (self.cards[0][0], self.cards[0][1])

        # Oszukiwanie z pewnym prawdopodobieństwem
        if self.cards[0][0] < declared_card[0]:
            if np.random.choice([True, False], p=[self.shouldCheat, 1 - self.shouldCheat]):
                self.cheat = True

        # nie mam karty, którą mogę zagrać i nie jest to ostatnia karta
        # w myśl tego, że dobieranie nie jest opłacalnym ruchem — muszę oszukać.
        if self.cards[-1][0] < declared_card[0]:
            self.cheat = True

        # oszustwo
        if self.cheat:
            # jak oszukiwać, to dam najmniejszą
            card = self.cards[0]
            if declared_card[0] < self.cards[-1][0]:
                known_cards = [card for card in self.cards if card[0] > declared_card[0]]
                say = np.random.choice(known_cards)
                declaration = (say[0], say[1])
            else:
                declaration = (min(declared_card[0] + 1, 14), np.random.choice([0, 1, 2, 3]))  # losowe kłamstwo
            return card, declaration

        # zwykły ruch
        else:
            if np.random.choice([True, False], p=[self.randomlyPlayTopCard, 1 - self.randomlyPlayTopCard]):
                # randomly play the top card
                card = self.cards[-1]
                declaration = (self.cards[-1][0], self.cards[-1][1])
                return card, declaration
            else:
                # play the lowest possible card
                for lookForaCard in self.cards:
                    if lookForaCard[0] < declared_card[0]:
                        continue
                    else:
                        card = lookForaCard
                        declaration = (lookForaCard[0], lookForaCard[1])
                        return card, declaration
        # print("HUUUUH-count?")  # for debug
        return "draw"  # never return "draw" :)) (do tego nigdy nie dojdzie)

    def checkCard(self, opponent_declaration):
        self.last_opp_card = opponent_declaration
        self.opponent_declarations.append(opponent_declaration)
        if opponent_declaration in self.cards:
            return True
        if opponent_declaration in self.card_pile:
            return True
        if opponent_declaration in self.opponent_declarations:
            return True  # to jest podejrzenie, pewnie nie zawsze będzie oszustwem, jeśli wcześniej mu się udało oszukać
        return np.random.choice([True, False], p=[self.checkProb, 1 - self.checkProb])

    # Notification sent at the end of a round
    # One may implement this method, capture data, and use it to get extra info
    # -- checked = TRUE -> someone checked. If FALSE, the remaining inputs do not play any role
    # -- iChecked = TRUE -> I decided to check my opponent (so it was my turn);
    #               FALSE -> my opponent checked, and it was his turn
    # -- iDrewCards = TRUE -> I drew cards (so I checked but was wrong or my opponent checked and was right);
    #                 FALSE -> otherwise
    # -- revealedCard - some card (X, Y). Only if I checked.
    # -- noTakenCards - number of taken cards
    def getCheckFeedback(self, checked, iChecked, iDrewCards, revealedCard, noTakenCards, log=True):
        self.roundsCounter += 1
        self.learningRate = 1 / self.roundsCounter
        # print("revealedCard: ", revealedCard) # karta - none - karta - none
        if checked:
            if noTakenCards > 0:
                self.card_pile = self.card_pile[:noTakenCards] if len(self.card_pile) > noTakenCards else []
            if iChecked:
                # I Checked
                if iDrewCards:
                    # I checked and I was wrong, so I should check less often
                    self.checkProb = max(0.01, self.checkProb - self.learningRate)
                else:
                    # I checked and I was right, so I should check more often (?)
                    self.checkProb = min(0.99, self.checkProb + self.learningRate)
            else:
                # My opponent checked, so I should cheat less often
                self.shouldCheat = max(0.01, self.shouldCheat - self.learningRate)

        else:
            # No one checked, so maybe I should cheat more often?
            self.shouldCheat = min(0.99, self.shouldCheat + self.learningRate)

# W ogólności to daje możliwości, pamiętania ile wie mój przeciwnik i jakie karty widział, etc.
# jeszcze tego nie zrobię, ale może się przydać,
