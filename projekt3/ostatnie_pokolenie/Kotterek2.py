import numpy as np
from player import Player


class Kotterek2(Player):

    def __init__(self, name):
        super().__init__(name)
        self.opponent_number_of_cards = 8
        self.checked = False
        self.pile = []

    def putCard(self, declared_card):
        if declared_card is not None:
            self.pile.append((-declared_card[0], declared_card[1]))
        if self.checked:
            self.checked = False
        elif declared_card is None:
            self.opponent_number_of_cards += min(3, len(self.pile))
            self.pile = self.pile[:-min(3, len(self.pile))]
        else:
            self.opponent_number_of_cards -= 1
        under = []
        over = []
        myCards = self.cards.copy()
        myCards.sort(key=lambda x: x[0])
        if declared_card is not None:
            for x in myCards:
                if x[0] < declared_card[0]:
                    under.append(x)
                else:
                    over.append(x)
        else:
            over = myCards
        if len(under) == 0:
            return over[0], over[0]
        if len(self.cards) == 1 and len(over) == 0:
            return "draw"
        if len(over) == 0:
            return under[0], (declared_card[0], (declared_card[1] + 1) % 4)
        if float(len(over)) / len(under) <= 1:
            return under[0], over[0]
        return over[0], over[0]

    def checkCard(self, opponent_declaration):
        if opponent_declaration in self.cards or opponent_declaration in self.pile:
            return True  # Ja WIEM gdzie ta karta jest, oszuście!
        if self.opponent_number_of_cards <= 2:
            return True  # Za mało masz, weź jeszcze!
        if self.opponent_number_of_cards <= 4 and len(self.cards) > self.opponent_number_of_cards:
            return True  # Mało masz, a co gorsza mniej niż ja!
        return False

    def getCheckFeedback(self, checked, iChecked, iDrewCards, revealedCard, noTakenCards, log=True):
        if checked and not iDrewCards:
            self.opponent_number_of_cards += noTakenCards - 1
            self.pile = self.pile[:-(noTakenCards - 1)]
            self.checked = True
        elif checked and iDrewCards:
            self.pile = self.pile[:-noTakenCards]