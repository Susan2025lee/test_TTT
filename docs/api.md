# Tic Tac Toe Agent API Documentation

## Overview

This document describes the API for the Tic Tac Toe agent system, including all major components and their interactions.

## Components

### 1. Game Environment (`GameEnvironment`)

The game environment manages the game state and rules.

#### Methods

- `reset() -> np.ndarray`: Reset the game to initial state
- `step(action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict]`: Take a game step
- `get_state() -> np.ndarray`: Get current game state
- `get_valid_moves() -> List[Tuple[int, int]]`: Get list of valid moves
- `is_valid_move(action: Tuple[int, int]) -> bool`: Check move validity
- `get_current_player() -> int`: Get current player (1 or -1)
- `is_game_over() -> bool`: Check if game is over
- `get_winner() -> Optional[int]`: Get game winner
- `make_move(move: Tuple[int, int]) -> bool`: Make a move

### 2. Agent Core (`Agent`)

The main agent class that coordinates decision making.

#### Methods

- `select_move(board: Board) -> Tuple[int, int]`: Select next move
- `update_from_game_result(outcome: int)`: Process game outcome
- `reset()`: Reset agent state

### 3. Memory System

#### Short-term Memory (`ShortTermMemory`)

Manages current game information.

- `add_state(state: Board)`: Add board state
- `add_move(move: Tuple[int, int])`: Add move
- `get_last_move() -> Optional[Tuple[int, int]]`: Get most recent move
- `get_current_sequence() -> List[Board]`: Get current game sequence

#### Experience Memory (`ExperienceMemory`)

Stores long-term game experiences.

- `add_game(states: List[Board], outcome: int)`: Add completed game
- `get_similar_states(state: Board) -> List[Board]`: Find similar states
- `get_winning_moves(state: Board) -> List[Tuple[int, int]]`: Get successful moves

#### Strategic Memory (`StrategicMemory`)

Stores strategic patterns and knowledge.

- `add_pattern(pattern: str, value: float)`: Add strategic pattern
- `get_pattern_value(pattern: str) -> float`: Get pattern value
- `update_pattern(pattern: str, outcome: int)`: Update pattern value

### 4. Training System (`SelfPlayTrainer`)

Manages agent training through self-play.

#### Methods

- `train() -> Dict`: Run training process
- `_evaluate_performance(game_num: int)`: Evaluate agent performance

#### Training Configuration

- `num_games`: Number of training games
- `eval_frequency`: Evaluation frequency
- `metrics`: Performance tracking

### 5. Learning System

#### State Evaluator (`StateEvaluator`)

Evaluates board states and features.

- `evaluate_state(state: Board) -> float`: Get state value
- `extract_features(state: Board) -> Dict`: Get state features

#### Action Selector (`ActionSelector`)

Selects moves based on evaluation and strategy.

- `select_action(state: Board, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]`: Choose next move
- `get_move_probabilities(state: Board) -> Dict[Tuple[int, int], float]`: Get move probabilities

## Usage Examples

### Basic Game Play
```python
from tic_tac_toe.environment import GameEnvironment
from tic_tac_toe.agent import Agent

# Create environment and agent
env = GameEnvironment()
agent = Agent(player_id=1)

# Play a game
state = env.reset()
while not env.is_game_over():
    move = agent.select_move(env.get_board())
    state, reward, done, info = env.step(move)
```

### Training an Agent
```python
from tic_tac_toe.training import SelfPlayTrainer

# Create trainer
trainer = SelfPlayTrainer(agent, num_games=1000)

# Run training
metrics = trainer.train()

# Check performance
print(f"Final win rate: {metrics['win_rates'][-1]}")
```

### Using Memory Systems
```python
from tic_tac_toe.memory import MemorySystem

# Create memory system
memory = MemorySystem()

# Add game experience
memory.short_term.add_state(current_board)
memory.experience.add_game(game_states, outcome)
memory.strategic.add_pattern(pattern, value)
```

## Error Handling

The system uses the following error handling approaches:

1. Invalid moves return False or None
2. Invalid states raise ValueError
3. Configuration errors raise ValueError
4. Type mismatches raise TypeError

## Performance Considerations

1. State evaluation is cached for efficiency
2. Pattern matching uses optimized string comparison
3. Memory systems use priority-based storage
4. Training can be distributed across multiple processes 