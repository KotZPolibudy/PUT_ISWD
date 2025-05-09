import numpy as np
from player import Player


class Kotterek(Player):

    def __init__(self, name):
        super().__init__(name)
        self.pile = []
        self.opponent_number_of_cards = 8
        self.moves = 0
        self.checked = False
        self.deck = []

    def putCard(self, declared_card):
        self.moves += 1
        if declared_card is not None:
            self.deck.append((-declared_card[0], declared_card[1]))
        if self.checked:
            self.checked = False
        elif declared_card is None:
            self.opponent_number_of_cards += min(3, len(self.deck))
            self.deck = self.deck[:-min(3, len(self.deck))]
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
        # if self.opponent_number_of_cards == 1:
        #     return myCards[-1], (max(myCards[-1][0], declared_card[0]), myCards[-1][1])
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
        if opponent_declaration in self.cards or opponent_declaration in self.deck:
            return True  # Ja WIEM gdzie ta karta jest, oszuście!
        if self.opponent_number_of_cards <= 2:
            return True  # Za mało masz, weź jeszcze!
        return False

    def getCheckFeedback(self, checked, iChecked, iDrewCards, revealedCard, noTakenCards, log=True):
        if checked and not iDrewCards:
            self.opponent_number_of_cards += noTakenCards - 1
            self.deck = self.deck[:-(noTakenCards - 1)]
            self.checked = True
        elif checked and iDrewCards:
            self.deck = self.deck[:-noTakenCards]