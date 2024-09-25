import numpy as np
import random
import utils
from consts import *

            
class Player:
    def __init__(self):
        self.hand = None
        self.moveset = None
        self.landlord = False
        self.free = False
                   
            
class DefaultPlayer(Player):
    def __init__(self):
        super().__init__()
        
        
    def move(self, pattern=None, prev_choice=None, leading_rank=-1):
        """
        Make a random valid move based on the current state of the game.
        
        params:
            pattern      (str):  The pattern of the previous player
            prev_choice  (array): The cards played by the previous player
            leading_rank (int):  The rank of the leading card
            
        returns:
            valid_move   (bool): Whether the player made a valid move
            pattern      (str):  The pattern of the move
            choice       (array): The cards played by the player
            leading_rank (int):  The rank of the leading card
            remainder    (int):  Number of cards remaining in the player's hand
        """
        # If free to move, play the smallest available hand for a random available pattern
        if self.free:
            leading_rank = -1   # Reset the leading rank
            self.free = False
            random.shuffle(self.moveset)
            for pattern in self.moveset:
                valid_move, pattern, choice, leading_rank = utils.smallest_valid_choice(hand=self.hand, pattern=pattern)
                if valid_move:
                    break
        
        # Else follow the pattern of the player before it and play a higher rank
        else:
            valid_move, pattern, choice, leading_rank = utils.smallest_valid_choice(hand=self.hand, pattern=pattern, leading_rank=leading_rank)
                
        # Return the card choice and subtract it from its hand
        if valid_move:
            choice = np.array(choice)
            self.hand -= choice
        return valid_move, pattern, choice, leading_rank, np.sum(self.hand)
    
    
    def claim_landlord(self, cards):
        """
        Randomly decide whether to claim the landlord cards.
        
        params:
            cards (array): The remaining cards in the deck [Unused]
            
        returns:
            landlord (bool): Whether the player claims the landlord cards
        """
        self.landlord = random.choice([True, False])
        return self.landlord
       
    
class UserPlayer(Player):
    def __init__(self):
        super().__init__()
        

    # The first card is the leading rank
    def move(self, pattern=None, prev_choice=None, leading_rank=-1):
        """
        Asks the user to input a move based on the current state of the game.
        
        params:
            pattern      (str):  The pattern of the previous player
            prev_choice  (array): The cards played by the previous player
            leading_rank (int):  The rank of the leading card
            
        returns:
            valid_move      (bool): Whether the player made a valid move
            pattern         (str):  The pattern of the move
            choice          (array): The cards played by the player
            leading_rank    (int):  The rank of the leading card
            remainder       (int):  Number of cards remaining in the player's hand
        """
        # Get the user input
        print(f"Hand: {utils.freq_array_to_card_str(self.hand)}")
        
        while True:
            valid_input = True
            if self.free:
                leading_rank = -1   # Reset the leading rank
                print("FREE TO MOVE")
                pattern = input("Enter the pattern: ")  # 1x5 format
                
                # Check if the pattern exists in the moveset
                if not pattern in self.moveset:
                    print("Invalid pattern. Please try again.")
                    continue
                
            # Assumes the first card is the leading rank
            user_cards = input("Enter your move: ")    # 334455 format
            
            # Check if all cards are known in the card set
            for c in user_cards:
                if c not in CARDS.values():
                    valid_input = False
            
            if valid_input:
                valid_move, pattern, choice, user_rank, valid_input = utils.read_user_cards(user_cards, pattern, leading_rank, self.hand)   # Convert to numpy frequency array
                
            # Escape the while loop only if the input is valid
            if not valid_input:
                print("Invalid card selection. Please try again.")
            else:
                break
        
        # After a successful move, the player is no longer free to move
        self.free = False
            
        # Record the play
        if valid_move:
            choice = np.array(choice)
            self.hand -= choice
            leading_rank = user_rank
        return valid_move, pattern, choice, leading_rank, np.sum(self.hand)
    

    def claim_landlord(self, cards):
        """
        Lets the user choose whether to claim the landlord cards.
        
        params:
            cards (array): The remaining cards in the deck
            
        returns:
            landlord (bool): Whether the player claims the landlord cards
        """
        print(f"\nCards in hand: {utils.freq_array_to_card_str(self.hand)}")
        print(f"Landlord cards: {utils.freq_array_to_card_str(cards)}")
        self.landlord = input("Claim the landlord cards? (y/n): ") == "y"
        return self.landlord
        
           
# TODO: Implement the RL player
class RLPlayer(Player):
    def __init__(self):
        super().__init__()
        
        
    def move(self, pattern=None, prev_choice=None, leading_rank=-1):
        pass
    
    
    def claim_landlord(self, cards):
        pass
 