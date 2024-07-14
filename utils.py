from consts import *

# Choose the smallest playable hand given a pattern
def smallest_valid_choice(hand, pattern, prev_choice=None, leading_rank=-1):
    """
    params:
        hand: Numpy frequency array like [1 1 1 3 0 1 1 2 0 2 1 0 1 0 0]
        pattern: String name of the pattern like "3+2"
        
    returns:
        contains_pattern: True or False for whether the hand contains the given pattern
        choice: The smallest set of cards that satisfy the pattern
    """
    contains_pattern = False
    choice = [0] * 15
    
    match pattern:
        # Lead is the lowest rank
        case "1x5":
            singles_count = 0
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 1:
                    singles_count += 1
                else:
                    singles_count = 0
                    
                if singles_count == 5:
                    contains_pattern = True
                    for i in range(5):
                        choice[rank-i] = 1
                    leading_rank = rank - 4
                    break
        
        # Lead is the lowest rank
        case "2x3":
            doubles_count = 0
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 2:
                    doubles_count += 1
                else:
                    doubles_count = 0
                    
                if doubles_count == 3:
                    contains_pattern = True
                    for i in range(3):
                        choice[rank-i] = 2
                    leading_rank = rank - 2
                    break
                
        case "3+1":
            triple_rank = -1
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 3:
                    triple_rank = rank
                    break
            if triple_rank >= 0:
                for rank, freq in enumerate(hand):
                    if freq >= 1 and rank != triple_rank:
                        contains_pattern = True
                        choice[triple_rank] = 3
                        choice[rank] = 1
                        leading_rank = triple_rank
                        break
        
        case "3+2":
            triple_rank = -1
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 3:
                    triple_rank = rank
                    break
            if triple_rank >= 0:
                for rank, freq in enumerate(hand):
                    if freq >= 2 and rank != triple_rank:
                        contains_pattern = True
                        choice[triple_rank] = 3
                        choice[rank] = 2
                        leading_rank = triple_rank
                        break
        
        case "1":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 1:
                    contains_pattern = True
                    choice[rank] = 1
                    leading_rank = rank
                    break
        
        case "2":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 2:
                    contains_pattern = True
                    choice[rank] = 2
                    leading_rank = rank
                    break
        
        case "3":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 3:
                    contains_pattern = True
                    choice[rank] = 3
                    leading_rank = rank
                    break
                
        case "4":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 4:
                    contains_pattern = True
                    choice[rank] = 4
                    leading_rank = rank
                    break
        
        case "5":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 5:
                    contains_pattern = True
                    choice[rank] = 5
                    leading_rank = rank
                    break
        
        case "6":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 6:
                    contains_pattern = True
                    choice[rank] = 6
                    leading_rank = rank
                    break
        
        case "7":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 7:
                    contains_pattern = True
                    choice[rank] = 7
                    leading_rank = rank
                    break
                
        case "8":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 8:
                    contains_pattern = True
                    choice[rank] = 8
                    leading_rank = rank
                    break
                
        # Small bomb, handle later with card bomb
        case "4.5":
            pass
        
        # Big bomb, must skip turn
        case "8.5":
            pass
            
        case _:
            raise NotImplementedError("Pattern not available")
        
    # If can't play any moves, check for card bombs
    if not contains_pattern:
        # If the previous pattern was not a bomb, any bomb can be played
        if pattern not in ["4", "5", "6", "7", "8", "4.5", "8.5"]:
            for rank, freq in enumerate(hand):
                if freq >= 4:
                    contains_pattern = True
                    choice[rank] = freq     # Bomb with all cards of that rank
                    leading_rank = rank
                    pattern = str(freq)     # ex. "5"
                    break
        
        # Else only bombs with bigger number or more cards can be played
        else:
            for rank, freq in enumerate(hand):
                if (freq == float(pattern) and rank > leading_rank) or freq > float(pattern):
                    contains_pattern = True
                    choice[rank] = freq
                    leading_rank = rank
                    pattern = str(freq)
                    break
                
    # If there are no card bombs, check for Joker bombs
    if not contains_pattern:
        # Small Joker bomb can only override regular cards and 4-bombs
        if pattern not in ["5", "6", "7", "8", "4.5"]:
            if hand[13] == 1 and hand[14] == 1:
                contains_pattern = True
                choice[13] = 1
                choice[14] = 1
                leading_rank = 13
                pattern = "4.5"
                
        # Big Joker bomb can override all cards and bombs
        else:
            if hand[13] == 2 and hand[14] == 2:
                contains_pattern = True
                choice[13] = 2
                choice[14] = 2
                leading_rank = 13
                pattern = "8.5"
    
    return contains_pattern, pattern, choice, leading_rank

# Convert frequency array to card string
def write_user_cards(hand):
    card_str = ""
    for rank, freq in enumerate(hand):
        card_str += CARDS[rank] * freq
    return card_str
    
# Convert card string to frequency array
def read_user_cards(user_cards, pattern):
    if not user_cards:
        return False, pattern, None, -1
    else:
        choice = [0] * 15
        # Check if the user played a bomb then update pattern and choice
        if user_cards[0] == user_cards[-1]:
            pattern = str(len(user_cards))
            choice[RANKS[user_cards[0]]] = len(user_cards)
        elif user_cards == "SL":
            pattern = "4.5"
            choice[13] = 1
            choice[14] = 1
        elif user_cards == "SSLL":
            pattern = "8.5"
            choice[13] = 2
            choice[14] = 2
        # Else keep following the previous pattern
        else:
            for c in user_cards:
                choice[RANKS[c]] += 1
        return True, pattern, choice, RANKS[user_cards[0]]