import numpy as np
from player import Player


class KociakProba2(Player):
    def __init__(self, name):

        super().__init__(name)
        self.shouldCheat = 0.3
        self.knownCards = set()
        self.checkProb = 0.2
        self.roundCounter = 1
        self.lr = 0.5

    def putCard(self, declared_card):
        self.cards.sort(key=lambda x: x[0])
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
                    declaration = np.random.randint(declared_card[0], 15), np.random.randint(0, 4)
                    i = 0
                    while declaration in self.knownCards:
                        i += 1
                        declaration = np.random.randint(declared_card[0], 15), np.random.randint(0, 4)  # try again
                        if i == 100:
                            break
                    return minCard, declaration
                else:
                    # losowy cheat
                    if np.random.choice([True, False], p=[self.shouldCheat, 1 - self.shouldCheat]):
                        # TU SIE ROZNI OD _proba
                        declaration = np.random.randint(declared_card[0], 15), np.random.randint(0, 4)
                        i = 0
                        while declaration in self.knownCards:
                            i += 1
                            declaration = np.random.randint(declared_card[0], 15), np.random.randint(0, 4)  # try again
                            if i == 100:
                                break
                        return minCard, declaration

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
        elif opponent_declaration in self.knownCards:  # to dość złudne, chyba jednak nie 100% false...
            return False
        elif opponent_declaration not in self.knownCards:
            # PARAMETR DO USTAWIENIA x/16 kart w grze
            l = len(self.knownCards)
            if l >= 15:
                return True
            elif l > 13:
                self.checkProb = 0.8
            elif l > 12:
                self.checkProb = 0.5
            elif l > 10:
                self.checkProb = 0.3
            else:
                self.checkProb = 0.1
            return np.random.choice([True, False], p=[self.checkProb, 1 - self.checkProb])

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
        if revealedCard is not None:
            self.knownCards.add(revealedCard)
        self.knownCards.update(self.cards)
        self.roundCounter += 1
        self.lr = 1/self.roundCounter
