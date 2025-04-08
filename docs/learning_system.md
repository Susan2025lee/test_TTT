# Learning System Documentation

The learning system is the core component responsible for enabling the AI agent to understand the game state, choose moves effectively, and improve its strategy over time based on past experiences. It comprises several key modules:

1.  **State Evaluation (`learning/state_eval.py`)**: Analyzes the current board state to assess its strategic value.
2.  **Action Selection (`learning/action_selection.py`)**: Determines which move to make based on the state evaluation and exploration/exploitation strategies.
3.  **Experience Processing (`learning/experience_processor.py`)**: Processes the results of completed games to update the agent's knowledge and memory.

---

## 1. State Evaluation (`StateEvaluator`)

The `StateEvaluator` class is responsible for analyzing a given Tic Tac Toe board state and extracting meaningful features to understand the current tactical and strategic situation.

**Key Functions:**

*   **Feature Extraction**: Identifies various patterns and configurations on the board, such as:
    *   Lines (rows, columns, diagonals) and their composition (number of own/opponent pieces, empty spaces).
    *   Potential threats (opponent has two in a row with the third space empty).
    *   Potential winning moves (agent has two in a row with the third space empty).
    *   Forks (opportunities to create two simultaneous threats).
    *   Center control, corner control, edge control.
*   **Position Assessment**: Assigns values to individual board positions based on their strategic importance (e.g., center > corners > edges).
*   **Threat Detection**: Explicitly identifies and quantifies immediate threats posed by the opponent.
*   **Strategic Evaluation**: Combines various features and heuristics to produce an overall assessment of the board state's value for the current player.

This evaluation forms the basis for the agent's decision-making process.

---

## 2. Action Selection (`ActionSelector`)

The `ActionSelector` class implements different policies for choosing the agent's next move from the list of available valid moves. It balances *exploitation* (choosing the move believed to be the best based on current knowledge) and *exploration* (trying different moves to potentially discover better strategies).

**Implemented Strategies:**

*   **Epsilon-Greedy**: With probability `epsilon`, selects a random valid move (exploration). With probability `1-epsilon`, selects the move with the highest estimated value (exploitation).
*   **Pattern-Based**: Prioritizes moves that complete known winning patterns stored in strategic memory.
*   **UCB (Upper Confidence Bound)**: Selects moves based on both their estimated value and the uncertainty associated with that estimate. (Note: Current implementation might be basic or require further refinement).
*   **Temperature Sampling (Softmax)**: Selects moves probabilistically based on their values, using a temperature parameter to control the degree of randomness. Higher temperatures lead to more exploration.

The choice of action selection strategy significantly impacts the agent's learning speed and performance.

---

## 3. Experience Processing (`ExperienceProcessor`)

The `ExperienceProcessor` class is activated after a game concludes. It takes the sequence of states and moves from the completed game (stored in short-term memory) and the final outcome (win, loss, or draw) to update the agent's long-term knowledge.

**Key Functions:**

*   **Game Analysis**: Reviews the sequence of moves and states played during the game.
*   **Credit Assignment**: Attributes the final outcome back to the moves and states that led to it. This typically involves calculating discounted returns for each state-action pair.
*   **Memory Updates**: Updates the `ExperienceMemory` with the processed game data (states, actions, outcomes, calculated returns) and the `StrategicMemory` with newly identified winning/losing patterns or updates to position values.
*   **Pattern Learning**: Extracts and reinforces patterns associated with wins or losses, updating their values in `StrategicMemory`.

Through experience processing, the agent learns from its successes and failures, gradually refining its state evaluations and action selection policies. 