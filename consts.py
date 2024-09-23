NUM_RANKS = 15

CARDS = {0: "3",
         1: "4",
         2: "5",
         3: "6",
         4: "7",
         5: "8",
         6: "9",
         7: "0",
         8: "J",
         9: "Q",
         10: "K",
         11: "A",
         12: "2",
         13: "S",   # Small Joker
         14: "L"}   # Large Joker

RANKS = {"3": 0,
         "4": 1,
         "5": 2,
         "6": 3,
         "7": 4,
         "8": 5,
         "9": 6,
         "0": 7,
         "J": 8,
         "Q": 9,
         "K": 10,
         "A": 11,
         "2": 12,
         "S": 13,   # Small Joker
         "L": 14}   # Large Joker

BOMB_SET = ["4", "5", "6", "7", "8", "4.5", "8.5"]

# In each deck there are 4 of each card, small joker, and large joker
CARD_FREQ = [4] * 13 + [1, 1]

# Number of cards to deal to each player depending on the number of players and decks
CARDS_PER_PLAYER = {"indv": {4: {1: [14, 14, 13, 13],
                                 2: [27, 27, 27, 27]},
                             3: {1: [18, 18, 18],
                                 2: [36, 36, 36]}},
                    "lord": {3: {1: ([17, 17, 17], 3),    # 17 cards each and 3 to the landlord
                                 2: ([34, 34, 34], 6)}}}  # 34 cards each and 6 to the landlord

# RL reward rates
REWARDS = {"win": 10,
           "loss": -10,
           "valid": 0.1,
           "pass": -0.1}

# Simplified moveset
MOVESET_1 = ["1x5",
             
             "2x3",
             
             "3x1+1",
             "3x1+2",
             
             "1",
             "2",
             "3",
             "4",
             "5",
             "6",
             "7",
             "8",
             "4.5", # Small Joker bomb
             "8.5"] # Large Joker bomb

# Comprehensive moveset
MOVESET_2 = ["1x5",     # 5 Consecutive Singles
             "1x6",     # 6 Consecutive Singles
             "1x7",     # 7 Consecutive Singles
             "1x8",     # 8 Consecutive Singles
             "1x9",     # 9 Consecutive Singles
             "1x10",    # 10 Consecutive Singles
             "1x11",    # 11 Consecutive Singles
             "1x12",    # 12 Consecutive Singles
             "1x13",    # 13 Consecutive Singles
             "1x14",    # 14 Consecutive Singles
             "1x15",    # 15 Consecutive Singles
             
             "2x3",     # 3 Consecutive Pairs
             "2x4",     # 4 Consecutive Pairs
             "2x5",     # 5 Consecutive Pairs
             "2x6",     # 6 Consecutive Pairs
             "2x7",     # 7 Consecutive Pairs
             "2x8",     # 8 Consecutive Pairs
             "2x9",     # 9 Consecutive Pairs
             "2x10",    # 10 Consecutive Pairs
             "2x11",    # 11 Consecutive Pairs
             "2x12",    # 12 Consecutive Pairs
             "2x13",    # 13 Consecutive Pairs
             "2x14",    # 14 Consecutive Pairs
             "2x15",    # 15 Consecutive Pairs
             
             "3x1+1",   # 1 Triple + 1 Single
             "3x2+1",   # 2 Consecutive Triples + 2 Singles
            #  "3x3+1",   # 3 Consecutive Triples + 3 Singles
            #  "3x4+1",   # 4 Consecutive Triples + 4 Singles
            #  "3x5+1",   # 5 Consecutive Triples + 5 Singles
            #  "3x6+1",   # 6 Consecutive Triples + 6 Singles
             
             "3x1+2",   # 1 Triple + 1 Pair
             "3x2+2",   # 2 Consecutive Triples + 2 Pairs
            #  "3x3+2",   # 3 Consecutive Triples + 3 Pairs
            #  "3x4+2",   # 4 Consecutive Triples + 4 Pairs
            #  "3x5+2",   # 5 Consecutive Triples + 5 Pairs
             
             "1",       # Single
             "2",       # Pair
             "3",       # Triple
             
             "4",       # 4-Bomb
             "5",       # 5-Bomb
             "6",       # 6-Bomb
             "7",       # 7-Bomb
             "8",       # 8-Bomb
             "4.5",     # Small Joker bomb
             "8.5"]     # Large Joker bomb

