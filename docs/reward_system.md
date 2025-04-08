# Reward System Documentation

## Overview

The reward system in the Tic Tac Toe AI agent is designed to provide feedback to the agent during and after games, guiding its learning process. It uses a combination of immediate, terminal, and potentially shaped rewards (though shaped rewards might be implicitly handled through pattern values) to evaluate moves and game outcomes.

## Reward Components

### 1. Immediate Rewards

These rewards are given after each move and provide instant feedback on the tactical value of a move. While not explicitly defined as a separate "Reward System" class, these concepts are integrated into the learning and memory updates.

*   **Position-Based Value:** Moves are often evaluated based on the strategic importance of the board position (center > corners > edges). This is handled within `StrategicMemory`'s `get_position_value` and potentially influences move selection in the `Agent` core.
*   **Tactical Rewards (Implicit):**
    *   **Blocking Opponent:** Actions that block an opponent's potential win are highly valued. This is primarily handled in the `Agent._apply_strategy` method by checking for opponent wins and assigning high values to blocking moves.
    *   **Creating Opportunities (Setups):** Moves that create two-in-a-row patterns (setups) are positively reinforced. This is handled in `StrategicMemory.update_pattern_value` and `ExperienceProcessor.process_experience` by assigning higher values/updates to patterns like '110', '101', '011'.

### 2. Terminal Rewards

These rewards are given only at the end of the game and reflect the final outcome.

*   **Win:** A positive reward (typically +1.0) is assigned.
*   **Loss:** A negative reward (typically -1.0) is assigned.
*   **Draw:** A neutral reward (typically 0.0) is assigned.

These terminal rewards are the primary drivers for the value updates calculated in `ExperienceProcessor._calculate_returns` and used throughout the learning process (e.g., in `ExperienceProcessor.process_experience` and `StrategicMemory.add_game`).

### 3. Shaped Rewards (Implicit via Pattern Values)

While not a separate reward calculation, the concept of "shaping" rewards (guiding the agent towards intermediate goals) is achieved through the `StrategicMemory.pattern_values`.

*   **Progress Towards Winning:** Patterns representing progress (e.g., '110') are assigned positive values. When the agent makes a move creating such a pattern, its value is reinforced.
*   **Strategic Value:** The pattern values themselves act as a form of shaped reward, guiding the agent towards configurations that have historically led to positive outcomes. Updates in `update_pattern_value` reinforce valuable patterns.

## Implementation Details

*   **Reward Calculation:** Primarily occurs within the `ExperienceProcessor` when processing game results (`process_experience`, `_calculate_returns`) and within `StrategicMemory` when adding games (`add_game`) and updating pattern values (`update_pattern_value`).
*   **Integration:** Rewards (especially terminal outcomes) are used to calculate target values for learning (e.g., Q-values or state values) and to update the statistical value of patterns and potentially moves/positions in `StrategicMemory`.
*   **Influence:** The reward signals directly influence how the `StrategicMemory` updates pattern values and how the `ExperienceProcessor` assigns credit or blame to moves made during the game.

## Future Considerations

*   **Explicit Shaped Rewards:** A dedicated shaped reward function could be added to provide more granular feedback during the game, potentially accelerating learning.
*   **Reward Scaling:** Experimenting with different scales for terminal vs. immediate/shaped rewards might be beneficial.
*   **Dynamic Rewards:** Adjusting rewards based on the opponent's strength or game phase could be explored. 