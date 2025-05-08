import numpy as np
from player import Player
import random


class Janicki_Glowczewski_v2(Player):

    def __init__(self, name):
        super().__init__(name)
        self.numOpponentCards = None
        self.cards = []
        self.myCheatCards = []
        self.myCardsOnStack = []
        self.numCardsOnStack = 0
        self.myTurn = None

    def startGame(self, cards):
        self.cards = sorted(cards, key=lambda x: x[0])
        self.numOpponentCards = len(cards)
        self.stack = []
        self.numCardsOnStack = 8
        self.myTurn = True

    def takeCards(self, cards_to_take):
        self.cards = sorted(self.cards + cards_to_take, key=lambda x: x[0])

    def putCard(self, declared_card):

        self.myTurn = True

        # if starting from the beginning, play the lowest card
        if declared_card is None:
            put_card = self.cards[0]
            self.myCardsOnStack.append(put_card)
            return put_card, put_card

        legal_cards = [card for card in self.cards if card[0] >= declared_card[0]]
        should_bluff = len(self.cards) >= self.numOpponentCards + 4

        # cant cheat with the last card
        if len(self.cards) == 1:
            should_bluff = False

        # if there is no legal move, cheat with lowest card (or draw if cant cheat)
        if len(legal_cards) == 0:

            if len(self.cards) == 1:
                return "draw"

            put_card = self.cards[0]
            self.myCardsOnStack.append(put_card)
            # self.myCheatCards.append(put_card)
            colors = [0, 1, 2, 3]
            colors.remove(declared_card[1])
            declared_put_card = (declared_card[0], random.choice(colors))
            return put_card, declared_put_card

        # if i have a legal card, but i should bluff, play the lowest card and say it's the highest
        if should_bluff:
            put_card = legal_cards[0]
            self.myCardsOnStack.append(put_card)
            # self.myCheatCards.append(put_card)
            declared_put_card = legal_cards[-1]
            return put_card, declared_put_card
        # playing the lowest legal card
        else:
            put_card = legal_cards[0]
            self.myCardsOnStack.append(put_card)
            return put_card, put_card

    def checkCard(self, opponent_declaration):

        if opponent_declaration in self.cards:
            return True

        if opponent_declaration in self.myCardsOnStack:
            return True

        if len(self.cards) <= 2:
            return False

        for card in self.cards:
            if card[0] >= opponent_declaration[0]:
                return False

        # if opponent is close to winning, check more often
        if len(self.cards) > self.numOpponentCards + 2 and self.numOpponentCards <= 2:
            return random.random() < 0.6

        if len(self.cards) > self.numOpponentCards + 2:
            return random.random() < 0.2

        # randomly check
        return random.random() < 0.02

    def getCheckFeedback(
        self, checked, iChecked, iDrewCards, revealedCard, noTakenCards, log=True
    ):

        if log:
            print(
                "MyFeedback = "
                + self.name
                + " : checked this turn = "
                + str(checked)
                + "; I checked = "
                + str(iChecked)
                + "; I drew cards = "
                + str(iDrewCards)
                + "; revealed card = "
                + str(revealedCard)
                + "; number of taken cards = "
                + str(noTakenCards)
            )

        if self.myTurn == False:
            self.numOpponentCards -= 1

        self.numCardsOnStack += 1

        if noTakenCards:
            self.numCardsOnStack -= noTakenCards
            if iDrewCards:
                pass  # handled by game engine
                # potentially opponent know my revealed card
            else:
                self.numOpponentCards += noTakenCards
                # potentially i know opponent's revealed card

            if iChecked:
                self.myCardsOnStack = self.myCardsOnStack[:-1]
            else:
                self.myCardsOnStack = self.myCardsOnStack[:-2]

        # print(
        #     f"Num my cards: {len(self.cards)}, Num opponent cards: {self.numOpponentCards}"
        # )

        self.myTurn = False
