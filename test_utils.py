from utils import *
from consts import *

# Individual mode with 3 players
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
    
# Individual mode with 4 players
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
    
# Landlord mode with 3 players, landlord wins
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
    
# Landlord mode with 3 players, peasants win
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
def test_get_all_possible_moves():
    cards_remaining = np.array(CARD_FREQ) * 2
    all_moves = get_all_possible_moves(overwrite=True)
    deck_moves = deck_possible_moves(all_moves, cards_remaining, MOVESET_2)
    
    assert len(all_moves) == len(deck_moves)
    assert len(all_moves) == 3513

# Should filter down to exactly 2722 possible moves for 1 deck with moveset 2
def test_deck_possible_moves_1():
    cards_remaining = np.array(CARD_FREQ) * 1
    all_moves = get_all_possible_moves(overwrite=True)
    deck_moves = deck_possible_moves(all_moves, cards_remaining, MOVESET_2)
    
    assert len(deck_moves) == 2722
    
# Filtering with stricter requirements should yield less possible moves
def test_deck_possible_moves_2():
    cards_remaining_1 = np.array(CARD_FREQ) * 1
    cards_remaining_2 = np.array(CARD_FREQ) * 2
    all_moves = get_all_possible_moves(overwrite=True)
    
    deck_moves_2_2 = deck_possible_moves(all_moves, cards_remaining_2, MOVESET_2)
    deck_moves_1_2 = deck_possible_moves(all_moves, cards_remaining_1, MOVESET_2)
    deck_moves_2_1 = deck_possible_moves(all_moves, cards_remaining_2, MOVESET_1)
    deck_moves_1_1 = deck_possible_moves(all_moves, cards_remaining_1, MOVESET_1)
    
    assert len(deck_moves_2_2) > len(deck_moves_1_2)
    assert len(deck_moves_2_2) > len(deck_moves_2_1)
    
    assert len(deck_moves_1_2) > len(deck_moves_1_1)
    assert len(deck_moves_2_1) > len(deck_moves_1_1)

