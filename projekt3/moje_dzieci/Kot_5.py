import numpy as np
from player import Player


class Kot5(Player):
    def __init__(self, name):

        super().__init__(name)
        # startowe wartości
        self.shouldCheat = 0.2
        self.checkProb = 0.2
        self.cheat = False
        self.randomlyPlayTopCard = 0.02
        self.learningRate = 0.3
        self.roundsCounter = 0
        self.knownCards = set()

    def putCard(self, declared_card):
        self.cards.sort(key=lambda x: x[0])
        self.cheat = False
        self.knownCards.update(self.cards)  # dodaj swoje karty do ogólnego setu kart w grze

        # check if maybe i want to randomly cheat
        # check if i have to cheat if no legal moves other than draw available

        # NEVER DRAW - ale gra mówi, że ostatnia karta jest publiczna? huh?
        if len(self.cards) == 1 and declared_card is not None and self.cards[0][0] < declared_card[0]:
            return "draw"

        # to nie pierwszy ruch, ale nie mam karty, ktora moge zagrac
        # i nie jest to ostatnia karta
        if len(self.cards) != 1 and declared_card is not None and self.cards[0][0] < declared_card[0]:
            if np.random.choice([True, False], p=[self.shouldCheat, 1 - self.shouldCheat]):
                self.cheat = True

        # nie mam czego zagrac
        if len(self.cards) != 1 and declared_card is not None and self.cards[-1][0] < declared_card[0]:
            self.cheat = True

        # then play a card and declare a card
        if self.cheat:
            card = self.cards[0]
            possible_declarations = [c for c in self.cards if c[0] >= declared_card[0]]
            if declared_card[0] <= self.cards[-1][0]:
                # można też dać losową, ale wsm daje najmniejszą legitnie możliwą...
                declaration = min(possible_declarations, key=lambda x: x[0])
            else:
                declaration = (min(declared_card[0] + 1, 14), np.random.choice([0, 1, 2, 3]))  # losowe kłamstwo

        # Play legit move
        else:
            if np.random.choice([True, False], p=[self.randomlyPlayTopCard, 1 - self.randomlyPlayTopCard]):
                # randomly play the top card
                card = self.cards[-1]
                declaration = (self.cards[-1][0], self.cards[-1][1])
            else:
                # play the lowest possible card
                if declared_card is not None:
                    min_val = declared_card[0]
                    for lookForaCard in self.cards:
                        if lookForaCard[0] < min_val:
                            continue
                        else:
                            card = lookForaCard
                            declaration = (lookForaCard[0], lookForaCard[1])
                            break
                else:
                    # first turn - play the lowest card
                    card = self.cards[0]
                    declaration = (self.cards[0][0], self.cards[0][1])

        return card, declaration

    def checkCard(self, opponent_declaration):
        if opponent_declaration in self.cards:
            return True
        elif opponent_declaration in self.knownCards:
            return False
        elif opponent_declaration not in self.knownCards and len(self.knownCards) > 12:  # PARAMETR DO USTAWIENIA x/16 kart w grze
            return True  # sprawdzaj, jeśli znasz większość kart w grze a przeciwnik mówi że gra inną
        # to prawdopodobieństwo można uzależnić od ilości kart na ręce też - pomysł
        # co może mieć więcej sensu, jeśli już robię podejście pamiętające karty.
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
        self.roundsCounter += 1
        if revealedCard is not None:
            self.knownCards.add(revealedCard)
        self.learningRate = 1 / self.roundsCounter
        if checked:
            if iChecked:
                # I Checked
                if iDrewCards:
                    # I checked and I was wrong, so I should check less often
                    self.checkProb = max(0.01, self.checkProb - self.learningRate)
                else:
                    # I checked and I was right, so I should check more often (?)
                    self.checkProb = min(0.99, self.checkProb + self.learningRate)
            else:
                # My opponent checked
                # so I should cheat less often
                self.shouldCheat = max(0.01, self.shouldCheat - self.learningRate)

        else:
            # No one checked
            # so maybe I should cheat more often?
            self.shouldCheat = min(0.99, self.shouldCheat + self.learningRate)  # do dopasowania wartości

        # print(f"Round: {self.roundsCounter}, checkProb: {self.checkProb}, shouldCheat: {self.shouldCheat}")

