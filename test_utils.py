from utils import *
from consts import *

# Case 1: Individual mode with 3 players
def test_finalize_rewards_1():
    mode = "indv"
    num_players = 3
    winner = 0
    landlord = None
    
    # State, action, reward, next state, done
    episode_transitions = [({"self": {"id": 1}}, None, 1, {"self": {"id": 2}}, False),
                           ({"self": {"id": 2}}, None, 2, {"self": {"id": 0}}, True),
                           ({"self": {"id": 0}}, None, 1, {"self": {"id": 1}}, False),
                           ({"self": {"id": 1}}, None, 1, {"self": {"id": 2}}, False),
                           ({"self": {"id": 2}}, None, 1, {"self": {"id": 0}}, True)]
    
    expected_transitions = [({"self": {"id": 1}}, None, 1, {"self": {"id": 2}}, False),
                            ({"self": {"id": 2}}, None, 2, {"self": {"id": 0}}, True),
                            ({"self": {"id": 0}}, None, 20, {"self": {"id": 1}}, False),
                            ({"self": {"id": 1}}, None, -10, {"self": {"id": 2}}, False),
                            ({"self": {"id": 2}}, None, -10, {"self": {"id": 0}}, True)]
    
    finalize_rewards(mode, num_players, episode_transitions, winner, landlord)
    assert episode_transitions == expected_transitions
    
# Case 2: Individual mode with 4 players
def test_finalize_rewards_2():
    mode = "indv"
    num_players = 4
    winner = 1
    landlord = None
    
    # State, action, reward, next state, done
    episode_transitions = [({"self": {"id": 2}}, None, 1, {"self": {"id": 3}}, False),
                           ({"self": {"id": 3}}, None, 2, {"self": {"id": 0}}, True),
                           ({"self": {"id": 0}}, None, 1, {"self": {"id": 1}}, False),
                           ({"self": {"id": 1}}, None, 1, {"self": {"id": 2}}, False),
                           ({"self": {"id": 2}}, None, 1, {"self": {"id": 3}}, False),
                           ({"self": {"id": 3}}, None, 1, {"self": {"id": 0}}, True)]
    
    expected_transitions = [({"self": {"id": 2}}, None, 1, {"self": {"id": 3}}, False),
                            ({"self": {"id": 3}}, None, 2, {"self": {"id": 0}}, True),
                            ({"self": {"id": 0}}, None, -10, {"self": {"id": 1}}, False),
                            ({"self": {"id": 1}}, None, 30, {"self": {"id": 2}}, False),
                            ({"self": {"id": 2}}, None, -10, {"self": {"id": 3}}, False),
                            ({"self": {"id": 3}}, None, -10, {"self": {"id": 0}}, True)]
    
    finalize_rewards(mode, num_players, episode_transitions, winner, landlord)
    assert episode_transitions == expected_transitions
    
# Case 3: Landlord mode with 3 players, landlord wins
def test_finalize_rewards_3():
    mode = "lord"
    num_players = 3
    winner = 1
    landlord = 1
    
    # State, action, reward, next state, done
    episode_transitions = [({"self": {"id": 1}}, None, 1, {"self": {"id": 2}}, False),
                           ({"self": {"id": 2}}, None, 2, {"self": {"id": 0}}, True),
                           ({"self": {"id": 0}}, None, 1, {"self": {"id": 1}}, False),
                           ({"self": {"id": 1}}, None, 1, {"self": {"id": 2}}, False),
                           ({"self": {"id": 2}}, None, 1, {"self": {"id": 0}}, True)]
    
    expected_transitions = [({"self": {"id": 1}}, None, 1, {"self": {"id": 2}}, False),
                            ({"self": {"id": 2}}, None, 2, {"self": {"id": 0}}, True),
                            ({"self": {"id": 0}}, None, -10, {"self": {"id": 1}}, False),
                            ({"self": {"id": 1}}, None, 20, {"self": {"id": 2}}, False),
                            ({"self": {"id": 2}}, None, -10, {"self": {"id": 0}}, True)]
    
    finalize_rewards(mode, num_players, episode_transitions, winner, landlord)
    assert episode_transitions == expected_transitions
    
# Case 4: Landlord mode with 3 players, peasants win
def test_finalize_rewards_4():
    mode = "lord"
    num_players = 3
    winner = 0
    landlord = 1
    
    # State, action, reward, next state, done
    episode_transitions = [({"self": {"id": 1}}, None, 1, {"self": {"id": 2}}, False),
                           ({"self": {"id": 2}}, None, 2, {"self": {"id": 0}}, True),
                           ({"self": {"id": 0}}, None, 1, {"self": {"id": 1}}, False),
                           ({"self": {"id": 1}}, None, 1, {"self": {"id": 2}}, False),
                           ({"self": {"id": 2}}, None, 1, {"self": {"id": 0}}, True)]
    
    expected_transitions = [({"self": {"id": 1}}, None, 1, {"self": {"id": 2}}, False),
                            ({"self": {"id": 2}}, None, 2, {"self": {"id": 0}}, True),
                            ({"self": {"id": 0}}, None, 10, {"self": {"id": 1}}, False),
                            ({"self": {"id": 1}}, None, -20, {"self": {"id": 2}}, False),
                            ({"self": {"id": 2}}, None, 10, {"self": {"id": 0}}, True)]
    
    finalize_rewards(mode, num_players, episode_transitions, winner, landlord)
    assert episode_transitions == expected_transitions



# Should not generate any moves that are impossible for 2 decks with moveset 2
def test_get_all_moves():
    cards_remaining = np.array(CARD_FREQ) * 2
    all_moves = get_all_moves(overwrite=True)
    deck_moves = get_deck_moves(all_moves, cards_remaining, MOVESET_2)
    
    assert len(all_moves) == len(deck_moves)
    assert len(all_moves) == 3513



# Case 1: Should filter down to exactly 2722 possible moves for 1 deck with moveset 2
def test_deck_moves_1():
    cards_remaining = np.array(CARD_FREQ) * 1
    all_moves = get_all_moves(overwrite=True)
    deck_moves = get_deck_moves(all_moves, cards_remaining, MOVESET_2)
    
    assert len(deck_moves) == 2722
    
# Case 2: Filtering with stricter requirements should yield less possible moves
def test_deck_moves_2():
    cards_remaining_1 = np.array(CARD_FREQ) * 1
    cards_remaining_2 = np.array(CARD_FREQ) * 2
    all_moves = get_all_moves(overwrite=True)
    
    deck_moves_2_2 = get_deck_moves(all_moves, cards_remaining_2, MOVESET_2)
    deck_moves_1_2 = get_deck_moves(all_moves, cards_remaining_1, MOVESET_2)
    deck_moves_2_1 = get_deck_moves(all_moves, cards_remaining_2, MOVESET_1)
    deck_moves_1_1 = get_deck_moves(all_moves, cards_remaining_1, MOVESET_1)
    
    assert len(deck_moves_2_2) > len(deck_moves_1_2)
    assert len(deck_moves_2_2) > len(deck_moves_2_1)
    
    assert len(deck_moves_1_2) > len(deck_moves_1_1)
    assert len(deck_moves_2_1) > len(deck_moves_1_1)



# Case 1: Free to move, no previous pattern, all moves are valid
def test_get_hand_moves_1():
    # Assume that player hand is always a numpy array
    hand = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    deck_moves = [["1", 0, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  ["1", 1, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  ["1x5", 0, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    
    free = True
    prev_pattern = None
    prev_leading_rank = -1
    hand_mask = [True, True, True]

    valid_move, hand_mask, curr_mask = get_hand_moves(hand, free, prev_pattern, prev_leading_rank, hand_mask, deck_moves)
    
    assert valid_move == True
    assert hand_mask == [True, True, True]
    assert curr_mask == [True, True, True]

# Case 2: Free to move, ignore previous pattern, some moves are invalid
def test_get_hand_moves_2():
    # Assume that player hand is always a numpy array
    hand = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    deck_moves = [["1", 4, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  ["1x6", 0, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  ["2x3", 0, [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  ["2x3", 1, [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    
    free = True
    prev_pattern = "3x2+2"
    prev_leading_rank = 10
    hand_mask = [True, True, True, False]

    valid_move, hand_mask, curr_mask = get_hand_moves(hand, free, prev_pattern, prev_leading_rank, hand_mask, deck_moves)
    
    assert valid_move == True
    assert hand_mask == [True, False, False, False]
    assert curr_mask == [True, False, False, False]

# Case 3: Not free to move, but some moves are valid
def test_get_hand_moves_3():
    # Assume that player hand is always a numpy array
    hand = np.array([1, 1, 2, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    deck_moves = [["1", 4, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  ["1x6", 0, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  ["2x3", 1, [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  ["2x3", 2, [0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    
    free = False
    prev_pattern = "2x3"
    prev_leading_rank = 1
    hand_mask = [True, True, True, True]

    valid_move, hand_mask, curr_mask = get_hand_moves(hand, free, prev_pattern, prev_leading_rank, hand_mask, deck_moves)
    
    assert valid_move == True
    assert hand_mask == [True, True, False, True]
    assert curr_mask == [False, False, False, True]

# Case 4: Not free to move, and no playable moves
def test_get_hand_moves_4():
    # Assume that player hand is always a numpy array
    hand = np.array([1, 1, 2, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    deck_moves = [["1", 4, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  ["1x6", 0, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  ["2x3", 1, [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  ["2x3", 2, [0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    
    free = False
    prev_pattern = "2x3"
    prev_leading_rank = 2
    hand_mask = [True, True, True, True]

    valid_move, hand_mask, curr_mask = get_hand_moves(hand, free, prev_pattern, prev_leading_rank, hand_mask, deck_moves)
    
    assert valid_move == False
    assert hand_mask == [True, True, False, True]
    assert curr_mask == [False, False, False, False]