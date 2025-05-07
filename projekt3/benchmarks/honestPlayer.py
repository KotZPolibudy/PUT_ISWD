from player import Player
import numpy as np


class HonestPlayer(Player):

    def putCard(self, declared_card):

        # check if must draw
        if len(self.cards) == 1 and declared_card is not None and self.cards[0][0] < declared_card[0]:
            return "draw"

        self.cards.sort(key=lambda x: x[0])

        if declared_card is None:
            # play the lowest possible card
            card = self.cards[0]
            return card, (card[0], card[1])

        # honest player!
        if self.cards[-1][0] < declared_card[0]:
            return "draw"

        for card in self.cards:
            if card[0] >= declared_card[0]:
                return card, (card[0], card[1])

    def checkCard(self, opponent_declaration):
        if opponent_declaration in self.cards: return True
        return np.random.choice([True, False], p=[0.3, 0.7])