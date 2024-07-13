# TODOs
- Build a simplified game environment (simple rules and 3 players, individual win)
    x Reset the game (shuffle and deal the cards)
    x Display the game after each move
    x Encode movesets
    x Check if a moveset is valid
    x Display the game result
    x Build a naive player that follows a simple pattern but is able to complete its hand
        x Play the smallest valid moveset
    x Test the naive player for a few games
    - Allow the user to act as a player
    - Write another comprehensive moveset of patterns
    - Decide on a format for recording the game (and maybe annotate rewards later)
    - Ask ChatGPT to improve the code

- Decide on a model (probably PPO)
    - Embed the cards
    - Find a way to store and replay experiences
    - Self-play with the naive models and human player
    - Make sure the model is able to train and complete a game
    - Improve the model to have a good win rate

- Implement resetting and rule checking for other game modes
    - Test the edge cases

- Use the pre-trained simple model to continue training on the complex rule sets