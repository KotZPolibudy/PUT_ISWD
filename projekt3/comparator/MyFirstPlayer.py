import numpy as np
import random
from player import Player


class MyFirstPlayer(Player):
    checkProb = 0.2
    shouldCheat = 0.001
    cheat = False
    randomlyPlayTopCard = 0.02

    # player's random strategy
    def putCard(self, declared_card):
        # never return "draw" :))
        self.cards.sort(key=lambda x: x[0])
        self.cheat = False

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
            if declared_card[0] < self.cards[-1][0]:
                declaration = (
                    self.cards[-1][0], self.cards[-1][1])  # zmienic na randomowa z kart z reki, ktora > declared card
            else:
                # declaration = (self.cards[0][0], self.cards[0][1]) #TUUU był error
                declaration = (min(declared_card[0] + 1, 14), np.random.choice([0, 1, 2, 3]))  # losowe kłamstwo
                # return "draw" # wow, ale najgorzej, ale to chyba sytuacja gdzie przeciwnik
                # i tak widzi cały stosik, więc nie wiem czy jest sens kłamać.
                # (tho kłamałbym for the record jak ktoś nie sprawdza)

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

    ### randomly decides whether to check or not
    def checkCard(self, opponent_declaration):
        if opponent_declaration in self.cards:
            return True
        return np.random.choice([True, False], p=[self.checkProb, 1 - self.checkProb])
