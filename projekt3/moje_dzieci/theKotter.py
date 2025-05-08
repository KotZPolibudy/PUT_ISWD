import numpy as np
from player import Player


class Kotter(Player):

    def __init__(self, name):
        super().__init__(name)
        self.opponent_number_of_cards = 8
        self.pile = []
        self.checked_flag = False

    def putCard(self, declared_card):
        if declared_card is not None:  # przeciwnik zagrał
            self.pile.append((-declared_card[0], declared_card[1]))
        if self.checked_flag:
            self.checked_flag = False
        if not self.checked_flag and declared_card is None:  # przeciwnik zrobił draw
            self.opponent_number_of_cards += min(3, len(self.pile))
            self.pile = self.pile[:-min(3, len(self.pile))]
        else:  # przeciwnik zagrał kartę
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
        if opponent_declaration in self.cards:
            return True  # mam te kartę
        if self.opponent_number_of_cards <= 2:
            return True  # jego ostatnia karta
        if opponent_declaration in self.pile:
            return True  # nie masz jej, bo ją zagrałem!
        # if len(self.cards) + len(self.pile) >= 14: # warunek do poprawy czy w known_cards
        #     return False  # no dobra, możesz to mieć
        # if opponent_declaration[0] < 11:
        #     prob = 0.0
        # elif opponent_declaration[0] >= 12:
        #     prob = 0.5
        # else:
        #     prob = 0.3
        # return np.random.choice([True, False], p=[prob, 1 - prob])
        return False

    def getCheckFeedback(self, checked, iChecked, iDrewCards, revealedCard, noTakenCards, log=True):
        if checked and not iDrewCards:
            self.opponent_number_of_cards += noTakenCards - 1
            self.pile = self.pile[:-(noTakenCards - 1)]
            self.checked_flag = True
        elif checked and iDrewCards:
            self.pile = self.pile[:-noTakenCards]