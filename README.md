# K-in-a-Row Agent

This repository contains a Python implementation of agents for playing "K-in-a-Row with Forbidden Squares" and related games, including Tic-Tac-Toe, Five-in-a-Row, and Cassini. The project is designed for CSE 415 at the University of Washington.

## Main Agent: `natevitz_KInARow.py`
- **Authors:** Nathan Vitzthum and Grace Liu
- Implements an advanced agent for K-in-a-Row games.
- Features:
  - Minimax search with optional alpha-beta pruning for efficient move selection.
  - Move ordering based on static evaluation to improve search speed (see Extra Credit).
  - Tracks and reports statistics for each move: time taken, static evaluations, alpha-beta cutoffs, and Zobrist hashing stats.
  - Responds to special opponent remarks:
    - "Tell me how you did that": Gives a detailed explanation of the agent's decision process for the last move.
    - "What's your take on the game so far?": Provides a game summary and win prediction based on the current state.
  - Supports multiple game types (TTT, FIAR, Cassini) via the `game_types.py` module.

## How to Play
1. **Run the Game Master:**
   - Use `Game_Master_Offline.py` to run matches between agents.
   - Example: `python Game_Master_Offline.py`
2. **Choose Agents:**
   - The game master can run games between any two agents in the folder (e.g., `natevitz_KInARow.py`, `RandomPlayer.py`).
   - You can set up matches by editing the `test()` function in `Game_Master_Offline.py`.
3. **Supported Games:**
   - Tic-Tac-Toe (TTT)
   - Five-in-a-Row (FIAR)
   - Cassini (special variant)

## Folder Overview
- `natevitz_KInARow.py`: Main agent implementation (see features above).
- `RandomPlayer.py`: Baseline agent that selects moves randomly.
- `agent_base.py`: Base class for agents; provides the required interface.
- `game_types.py`: Defines game types and state representations.
- `winTesterForK.py`: Checks for win conditions after each move.
- `Game_Master_Offline.py`: Runs games between agents offline.
- `spec_static_by_table.py`: Special static evaluator for autograding.
- `autograder.py`: Automated tests for agent capabilities (static eval, minimax, etc.).
- `gameToHTML.py`: Generates HTML reports of games (if enabled).
- `ExtraCredit.txt`: Describes extra credit features and competition entry.
- Various image and transcript files: Used for game visualization and logging.

## Extra Credit Features
See `ExtraCredit.txt` for details. Highlights include:
- **Move Ordering:** The agent uses static evaluation to order moves during search, greatly improving efficiency, especially in late-game scenarios.
- **Detailed Explanations:** The agent responds to special remarks with explanations of its reasoning and game summaries.
- **Competition Entry:** The agent was entered into the class competition and tested against other agents.

## Getting Started
1. Ensure you have Python 3 installed.
2. Place all files in the same directory.
3. Run `Game_Master_Offline.py` to start a game.
4. Edit agent selection in `Game_Master_Offline.py` as needed.

## Contact
For questions or issues, contact the authors listed in `natevitz_KInARow.py`.
