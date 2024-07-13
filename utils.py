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
        
        case "2":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 2:
                    contains_pattern = True
                    choice[rank] = 2
                    leading_rank = rank
                    break
        
        case "1":
            for rank, freq in enumerate(hand):
                if rank > leading_rank and freq >= 1:
                    contains_pattern = True
                    choice[rank] = 1
                    leading_rank = rank
                    break
            
        case _:
            raise NotImplementedError("Pattern not available")
    
    return contains_pattern, choice, leading_rank