import numpy as np
from player import Player


# To jest ten bot o którym mówiłem że będzie rzucał stolikiem jak zacznie przegrywać.
# Sam jestem ciekaw tych wyników, chociaż wiem że będą poprzedzone spamem o errorach
class Rzucacz(Player):

    def __init__(self, name):
        super().__init__(name)
        self.enemy_cards = 8
        self.checked = False
        self.pile = []

    def putCard(self, declared_card):
        # round feedback update
        if declared_card is not None:
            self.pile.append(declared_card)
        if declared_card is None and not self.checked:
            self.enemy_cards += min(3, len(self.pile))
            self.pile = self.pile[:-min(3, len(self.pile))]
        else:
            self.enemy_cards -= 1

        if self.enemy_cards <= 2 or len(self.cards) > 10:
            # return self.cards[0], self.cards[0]  # to jest ten cheat
            return (2, 2), (2, 2)  # to jest ten cheat

        # putCard old
        lower = []
        legit = []
        self.cards.sort(key=lambda x: x[0])
        if declared_card is not None:
            for lookForaCard in self.cards:
                if lookForaCard[0] < declared_card[0]:
                    lower.append(lookForaCard)
                else:
                    legit.append(lookForaCard)
        else:
            return self.cards[0], self.cards[0]  # jak każda, to najniższa
        if len(self.cards) == 1 and len(legit) == 0:
            return "draw"  # jak muszę, to draw, ale tylko tutaj, bo draw jest zdominowane przez oszustwo
        if len(legit) == 0:
            return lower[0], (min(declared_card[0] + 1, 14), np.random.choice([0, 1, 2, 3]))  # trzeba kłamać, to będę
        if len(legit) - len(lower) < 0:
            return lower[0], legit[0]  # ej, ale trzeba się pozbyć tych niskich kart...
        return legit[0], legit[0]

    def checkCard(self, opponent_declaration):
        if opponent_declaration in self.cards or opponent_declaration in self.pile:
            return True  # Ja WIEM gdzie ta karta jest oszuście!
        if self.enemy_cards <= 2:
            return True  # Za mało masz, weź jeszcze!
        if self.enemy_cards <= 4 and len(self.cards) > self.enemy_cards:
            return True  # Mało masz, a co gorsza mniej niż ja!
        return False

    def getCheckFeedback(self, checked, iChecked, iDrewCards, revealedCard, noTakenCards, log=True):
        self.checked = False
        if checked and not iDrewCards:
            self.enemy_cards += noTakenCards - 1
            self.pile = self.pile[:-(noTakenCards - 1)]
            self.checked = True
        elif checked and iDrewCards:
            self.pile = self.pile[:-noTakenCards]
