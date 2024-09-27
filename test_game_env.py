import json

from game_env import GameEnv, TrainGameEnv

# Case 1: Able to run and record a game of 3 NPCs without error
def test_game_env_1():
    env = GameEnv(num_decks=2, num_players=3, mode="lord", players=[])
    env.reset()
    history = env.play_game(verbose=False)

    game_name = "data/user_game.json"
    json.dump(history, open(game_name, "w"))
    
    assert len(history) > 0

# Case 2: Run NPC games in various modes
def test_game_env_2():
    env = GameEnv(num_decks=1, num_players=3, mode="lord", players=[])
    env.reset()
    env.play_game(verbose=False)
    
    env = GameEnv(num_decks=2, num_players=3, mode="lord", players=[])
    env.reset()
    env.play_game(verbose=False)
    
    env = GameEnv(num_decks=1, num_players=3, mode="indv", players=[])
    env.reset()
    env.play_game(verbose=False)
    
    env = GameEnv(num_decks=1, num_players=4, mode="indv", players=[])
    env.reset()
    env.play_game(verbose=False)
    
    env = GameEnv(num_decks=2, num_players=3, mode="indv", players=[])
    env.reset()
    env.play_game(verbose=False)
    
    env = GameEnv(num_decks=2, num_players=4, mode="indv", players=[])
    env.reset()
    env.play_game(verbose=False)
    
    assert True