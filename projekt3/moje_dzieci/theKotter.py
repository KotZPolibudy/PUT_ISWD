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

        self.cards.sort(key=lambda x: x[0])
        if declared_card is None or self.cards[0][0] >= declared_card[0]:
            return self.cards[0], self.cards[0]  # wszystkie karty grają — zagraj najmniejszą.
        if len(self.cards) == 1 and self.cards[0][0] < declared_card[0]:  # ta druga czesc useless? -optim.
            return "draw"

        fake_declaration = (min(declared_card[0] + 1, 14), np.random.choice([0, 1, 2, 3]))
        if self.cards[-1][0] < declared_card[0]:
            return self.cards[0], fake_declaration
        else:
            nonLegitCount = 0
            legitCount = 0
            for lookForaCard in self.cards:
                if lookForaCard[0] < declared_card[0]:
                    nonLegitCount += 1
                    continue
                else:
                    fake_declaration = lookForaCard
                    legitCount = len(self.cards) - nonLegitCount
            if nonLegitCount - legitCount > 2:
                return self.cards[0], fake_declaration

        # legit ruch najmniejszej możliwej
        for lookForaCard in self.cards:
            if lookForaCard[0] >= declared_card[0]:
                return lookForaCard, lookForaCard

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