from utils import *

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


def test_generate_all_possible_moves():
    pass

