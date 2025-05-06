import numpy as np
from player import Player


class MySecondPlayerOptim(Player):

    def __init__(self, name):
        super().__init__(name)
        # startowe wartości
        self.shouldCheat = 0.2
        self.checkProb = 0.2
        self.cheat = False
        self.randomlyPlayTopCard = 0.02
        self.learningRate = 0.3
        self.roundsCounter = 0

    def putCard(self, declared_card):
        self.cards.sort(key=lambda x: x[0])
        self.cheat = False

        # pierwszy ruch
        if declared_card is None:
            return self.cards[0], (self.cards[0][0], self.cards[0][1])

        # ostatni ruch
        if len(self.cards) == 1:
            if self.cards[0][0] < declared_card[0]:
                return "draw"
            else:
                return self.cards[0], (self.cards[0][0], self.cards[0][1])

        # nie mam czego zagrac
        if self.cards[-1][0] < declared_card[0]:
            card = self.cards[0]
            if declared_card[0] < self.cards[-1][0]:
                # zmienic na randomowa z kart z reki, ktora > declared card i porownac
                declaration = (self.cards[-1][0], self.cards[-1][1])
            else:
                declaration = (min(declared_card[0] + 1, 14), np.random.choice([0, 1, 2, 3]))
            return card, declaration

        # zwykły ruch
        if self.cards[0][0] < declared_card[0]:
            if np.random.choice([True, False], p=[self.shouldCheat, 1 - self.shouldCheat]):
                card = self.cards[0]
                if declared_card[0] < self.cards[-1][0]:
                    # zmienic na randomowa z kart z reki, ktora > declared card i porownac
                    declaration = (self.cards[-1][0], self.cards[-1][1])
                else:
                    declaration = (min(declared_card[0] + 1, 14), np.random.choice([0, 1, 2, 3]))
                return card, declaration
            else:
                # legit move
                if np.random.choice([True, False], p=[self.randomlyPlayTopCard, 1 - self.randomlyPlayTopCard]):
                    # randomly play the top card
                    return self.cards[-1], (self.cards[-1][0], self.cards[-1][1])
                else:
                    # play the lowest possible card
                    min_val = declared_card[0]
                    for lookForaCard in self.cards:
                        if lookForaCard[0] < min_val:
                            continue
                        else:
                            return lookForaCard, (lookForaCard[0], lookForaCard[1])
        return "draw"  # never return draw ;)

    def checkCard(self, opponent_declaration):
        if opponent_declaration in self.cards:
            return True
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
        self.learningRate = 1 / self.roundsCounter
        if checked:
            if iChecked:
                if iDrewCards:
                    self.checkProb = max(0.01, self.checkProb - self.learningRate)
                else:
                    self.checkProb = min(0.99, self.checkProb + self.learningRate)
            else:
                self.shouldCheat = max(0.01, self.shouldCheat - self.learningRate)
        else:
            self.shouldCheat = min(0.99, self.shouldCheat + self.learningRate)  # do dopasowania wartości
