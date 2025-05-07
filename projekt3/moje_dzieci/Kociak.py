import numpy as np
from player import Player


class Kociak(Player):
    def __init__(self, name):

        super().__init__(name)
        # startowe wartości
        self.shouldCheat = 0.3
        self.checkProb = 0.2
        self.cheat = False
        self.randomlyPlayTopCard = 0.02
        self.learningRate = 0.3
        self.roundsCounter = 0
        self.knownCards = set()

    def putCard(self, declared_card):
        self.cards.sort(key=lambda x: x[0])
        self.cheat = False
        self.knownCards.update(self.cards)
        minCard = self.cards[0]

        if len(self.cards) == 1:
            if declared_card is None or self.cards[0][0] >= declared_card[0]:
                return minCard, minCard
            if declared_card is not None and self.cards[0][0] < declared_card[0]:
                return "draw"
        else:
            if declared_card is None:
                return minCard, minCard
            else:
                maxCard = self.cards[-1]
                if declared_card[0] > maxCard[0]:
                    # Trzeba oszukiwać
                    declaration = (min(declared_card[0] + 1, 14), np.random.choice([0, 1, 2, 3]))  # losowe kłamstwo
                    return minCard, declaration
                else:
                    # losowy cheat
                    if np.random.choice([True, False], p=[self.shouldCheat, 1 - self.shouldCheat]):
                        return minCard, maxCard  # chyba maxCard działa lepiej niż losowa z ręki, ale do sprawdzenia.

                # legit ruch
                # tu można dodać randomly play the top card, zobaczymy performance potem
                for lookForaCard in self.cards:
                    if lookForaCard[0] >= declared_card[0]:
                        return lookForaCard, lookForaCard
                print("WTF1")
            print("WTF2")
        print("WTF3")
        return "draw"

    def checkCard(self, opponent_declaration):
        if opponent_declaration in self.cards:
            return True
        elif opponent_declaration in self.knownCards:
            return False
        elif opponent_declaration not in self.knownCards and len(self.knownCards) > 14:
            # PARAMETR DO USTAWIENIA x/16 kart w grze
            # to by mozna na prawdopodobienstwie, w zaleznosci ile kart znam!
            return True  # sprawdzaj, jeśli znasz większość kart w grze a przeciwnik mówi że gra inną
        # to prawdopodobieństwo można uzależnić od ilości kart na ręce też - pomysł
        # co może mieć więcej sensu, jeśli już robię podejście pamiętające karty.
        # return np.random.choice([True, False], p=[self.checkProb, 1 - self.checkProb])
        return False

    # Notification sent at the end of a round
    # One may implement this method, capture data, and use it to get extra info
    # -- checked = TRUE -> someone checked. If FALSE, the remaining inputs do not play any role
    # -- iChecked = TRUE -> I decided to check my opponent (so it was my turn);
    #               FALSE -> my opponent checked and it was his turn
    # -- iDrewCards = TRUE -> I drew cards (so I checked but was wrong or my opponent checked and was right);
    #                 FALSE -> otherwise
    # -- revealedCard - some card (X, Y). Only if I checked.
    # -- noTakenCards - number of taken cards
    def getCheckFeedback(self, checked, iChecked, iDrewCards, revealedCard, noTakenCards, log=True):
        self.roundsCounter += 1
        if revealedCard is not None:
            self.knownCards.add(revealedCard)
        self.knownCards.update(self.cards)
