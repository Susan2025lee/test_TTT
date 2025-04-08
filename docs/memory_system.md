# Memory System Documentation

## Overview

The memory system is a crucial part of the Tic Tac Toe AI agent, enabling it to learn from past experiences and develop strategic understanding. It comprises three main components:

1.  **Short-Term Memory:** Holds information about the current game.
2.  **Experience Memory:** Stores complete games for later learning (replay).
3.  **Strategic Memory:** Manages long-term knowledge about patterns, openings, and counters.

These components are primarily implemented within the `tic_tac_toe/src/memory/` directory, with `StrategicMemory` (`strategic.py`) being the most complex.

## Components

### 1. Short-Term Memory

*   **Purpose:** To maintain the context of the ongoing game.
*   **Implementation:** Typically handled within the `Agent` class or the main game loop.
*   **Contents:**
    *   Current board state.
    *   Sequence of moves made so far in the current game.
    *   The opponent's last move (used for counter-move logic).
*   **Lifecycle:** Cleared at the start of each new game.

### 2. Experience Memory

*   **Purpose:** To store trajectories (sequences of states, actions, rewards) from completed games. This allows the agent to learn from batches of past experiences (experience replay).
*   **Implementation:** Could be a dedicated class (e.g., `ExperienceReplayBuffer`) or integrated within the `TrainingSystem` or `ExperienceProcessor`.
*   **Contents:**
    *   List of game states.
    *   List of moves (actions) taken.
    *   Final game outcome (win/loss/draw).
    *   Potentially intermediate rewards.
*   **Usage:** The `ExperienceProcessor` likely samples from this memory to update the agent's value functions or policies.

### 3. Strategic Memory (`memory/strategic.py`)

*   **Purpose:** To store and retrieve long-term strategic knowledge derived from many games.
*   **Implementation:** `StrategicMemory` class.
*   **Key Contents & Methods:**
    *   **`pattern_values` (Dict):** Stores the learned value (ranging from -1 to 1) associated with specific game patterns (e.g., `'row_110'`, `'diag_011'`). This is the core of the agent's pattern recognition.
        *   Updated by `update_pattern_value`.
        *   Queried by `get_pattern_value`.
    *   **Pattern Handling:**
        *   `_get_equivalent_patterns_non_recursive`: Finds rotated/reflected versions of a pattern string.
        *   `update_pattern_value`: Updates the value of a given pattern and its equivalents based on game outcomes and reinforcement signals. Includes logic for:
            *   Initialization of unseen patterns.
            *   Scaling updates based on pattern type (win, setup).
            *   Adaptive momentum (adjusting update size based on current value and change magnitude).
            *   Normalization (keeping values within [-1, 1]).
        *   `_update_related_patterns_non_recursive`: Propagates value updates from a primary pattern (e.g., a winning '111') to related sub-patterns (e.g., the '110' setup that led to it). This helps the agent learn precursor patterns.
    *   **Opening Moves (`opening_moves`, `opening_stats`):** Stores statistics (wins/total games) for different first moves. Potentially used to select strong opening moves (`get_best_opening`).
    *   **Counter Moves (`counter_moves`):** Stores statistics on the success rate of moves played immediately after a specific opponent move. Used to find effective counters (`get_best_counter`).
    *   **Position Values (`position_values`):** A simple 3x3 array holding static values for board positions (center, corners, edges), potentially updated slowly based on outcomes (`update_position_value`).
    *   **Game Integration (`add_game`):** Processes a completed game, updating opening stats, counter stats, and triggering pattern value updates based on the game's moves and outcome.
    *   **Move Evaluation Support:**
        *   `get_winning_moves`: Identifies immediate winning moves available from a given state by checking for patterns like '111' or high-value patterns.
        *   `get_strategic_value`: Provides a basic evaluation of a move based on position value and opening/winning potential (though the core agent likely uses more sophisticated evaluation).

## Data Flow & Interaction

1.  **During Game:** The agent uses `StrategicMemory` (e.g., `get_pattern_value`, `get_winning_moves`, potentially `get_best_counter`) and short-term memory to select moves.
2.  **After Game:** The completed game trajectory (states, moves, outcome) is potentially stored in `Experience Memory`.
3.  **Learning:** The `ExperienceProcessor` takes game data (either the last game or sampled from `Experience Memory`) and uses it to calculate value changes. These changes are then passed to `StrategicMemory.update_pattern_value` to refine the long-term pattern knowledge.
4.  **Statistics:** The `StrategicMemory.add_game` method directly updates opening and counter-move statistics after each game.

## Key Concepts

*   **Pattern Representation:** Patterns are stored as strings (e.g., `'row_101'`, `'col_022'`, `'diag_111'`, `'anti_diag_202'`) representing lines of 3 on the board from the perspective of player 1 (1=player 1, 2=player 2, 0=empty).
*   **Value Propagation:** Updating related patterns ensures that learning about a valuable pattern (like a win) also reinforces the patterns that typically lead to it.
*   **Momentum:** Helps stabilize learning by making larger updates when values are uncertain (near 0) or when a significant event occurs, and smaller updates when values are already strong (near +/- 1).
*   **Normalization:** Keeps pattern values bounded, preventing extreme values and potentially improving learning stability. 