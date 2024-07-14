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
    x Allow the user to act as a player
    x Write a comprehensive moveset of patterns
    x Test user player on simple moveset (Fix rank order when user passes)
    x Implement simple bombs
    - When bombing, consider the available moveset
    - Decide on a format for recording the game (and maybe annotate rewards later)
    - Ask ChatGPT to improve the code

- Decide on a model (probably PPO)
    - Embed the cards and movesets
    - Find a way to store and replay experiences
    - Self-play with the naive models and human player
    - Make sure the model is able to learn and complete a game
    - Improve the model to have a good win rate

- Implement more movesets
    - Write tests for moveset compatibility

- Add a smart player

- Use the pre-trained simple model to continue training on the complex rule sets