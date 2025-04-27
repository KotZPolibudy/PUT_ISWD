import numpy as np
from player import Player


class KotKilianczyk(Player):

    checkProb = 0.5
    should_cheat = 0.1
    cheat = False

    ### basic strategy
    def putCard(self, declared_card):
        self.cheat = np.random.random(1) < self.should_cheat

        if not self.cheat:
            return declared_card, declared_card
        else:
            # jak już oszukiwać, to wyrzuć najmniejszą karte
            return self.cards[0], declared_card

    ### randomly decides whether to check or not
    def checkCard(self, opponent_declaration):
        return np.random.random(1) < self.checkProb
