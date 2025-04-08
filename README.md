# Tic Tac Toe AI Agent System

A sophisticated AI agent system for playing Tic Tac Toe, featuring memory systems, reward functions, and learning capabilities. This project provides a complete game environment, an intelligent agent capable of learning through self-play, and tools for analyzing game patterns.

## Features

- 🎮 Complete Tic Tac Toe game environment
- 🧠 AI agent with multiple memory systems
- 📈 Learning and reward mechanisms
- 🔄 Self-play training capabilities
- 🎯 Strategic pattern recognition
- 📊 Performance tracking

## Documentation

Detailed documentation is available in the `docs/` directory:

*   **[User Guide](docs/user_guide.md)**: Overall guide for installation, setup, and usage examples.
*   **Component Documentation**:
    *   [Game Environment](docs/environment.md) (Placeholder)
    *   [Memory System](docs/memory_system.md) (Placeholder)
    *   [Reward System](docs/reward_system.md) (Placeholder)
    *   [Learning System](docs/learning_system.md)
    *   [Agent Core](docs/agentcore_system.md)
    *   [Training System](docs/training_system.md)
*   **[API Documentation](docs/api.md)**: Detailed API reference (Placeholder/To be generated).
*   **[Implementation Plan](implementation_plan.md)**: Original development roadmap.

## Project Structure

```
tic_tac_toe/
├── src/
│   ├── environment/     # Game environment and board state
│   ├── memory/         # Short-term, experience, and strategic memory
│   ├── reward/         # Reward calculation and game outcome processing
│   ├── learning/       # State evaluation and action selection
│   ├── agent/          # Core agent implementation
│   └── training/       # Self-play training system
├── tests/              # Test suite
├── docs/              # Documentation
└── examples/          # Usage examples
```

## Requirements

- Python 3.7+
- NumPy (>=1.21.0, installed automatically via pip)
- PyTest (for development/testing)
- Black, Flake8 (for development/linting)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [repository-url] # Replace with the actual URL
    cd your-repository-name    # Replace with the actual directory name
    ```

2.  **(Optional) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the package:**

    *   **For regular use:**
        ```bash
        pip install .
        ```
    *   **For development (editable install):**
        ```bash
        pip install -e .
        ```
    This command uses the `setup.py` file to install the `tic_tac_toe_agent` package and its required dependency (NumPy).

4.  **(Optional) Install development dependencies:**
    If you plan to run tests or contribute to development, install the development dependencies:
    ```bash
    pip install pytest pytest-cov black flake8
    ```

## Quick Start

```python
from tic_tac_toe.src.environment.board import Board
from tic_tac_toe.src.agent.core import Agent
from tic_tac_toe.src.memory.memory_system import MemorySystem
from tic_tac_toe.src.reward.reward_system import RewardSystem
from tic_tac_toe.src.learning.state_eval import StateEvaluator
from tic_tac_toe.src.learning.action_selection import ActionSelector
from tic_tac_toe.src.learning.experience_processor import ExperienceProcessor

# Initialize components
# Note: You might want a helper function or factory to create a pre-configured agent
def create_agent(player_id=1):
    memory = MemorySystem()
    reward = RewardSystem()
    evaluator = StateEvaluator()
    selector = ActionSelector()
    processor = ExperienceProcessor(memory.experience, memory.strategic, evaluator)
    agent = Agent(player_id=player_id,
                  memory_system=memory,
                  reward_system=reward,
                  state_evaluator=evaluator,
                  action_selector=selector,
                  experience_processor=processor)
    return agent

agent = create_agent(player_id=1)
board = Board()

# Game loop example (Agent vs Human)
print("Starting game: Agent (X) vs Human (O)")
while not board.is_game_over():
    print("\n" + str(board))
    if board.current_player == agent.player_id:
        print("Agent's turn...")
        move = agent.select_move(board)
        if move:
            print(f"Agent selects: {move}")
            board.make_move(move)
        else:
            print("Agent cannot find a move!") # Should not happen in standard play
            break
    else:
        print("Your turn (O).")
        valid_move = False
        while not valid_move:
            try:
                row = int(input("Enter row (0-2): "))
                col = int(input("Enter column (0-2): "))
                move = (row, col)
                if board.is_valid_move(move):
                    board.make_move(move)
                    valid_move = True
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Please enter numbers for row and column.")
            except IndexError:
                 print("Invalid input. Row/Col must be 0, 1, or 2.")

# Game finished
print("\nGame Over!")
print(board)
winner = board.check_winner()
if winner == 1:
    print("Agent (X) wins!")
elif winner == -1:
    print("You (O) win!")
else:
    print("It's a draw!")

# Update agent (optional, useful if agent is learning from this game)
# outcome = 1 if winner == agent.player_id else (-1 if winner == -agent.player_id else 0)
# agent.update_from_game_result(outcome)
```

## System Architecture

### 1. Game Environment (`environment/`)
- `board.py`: Game board representation and rules
- Core game mechanics and state management
- Move validation and win detection

### 2. Memory System (`memory/`)
- `memory_system.py`: Main memory system coordinator
- `short_term.py`: Current game state tracking
- `experience.py`: Long-term experience storage
- `strategic.py`: Pattern and strategy storage

### 3. Reward System (`reward/`)
- `reward_system.py`: Reward calculation and distribution
- Position-based rewards
- Strategic rewards
- Terminal state rewards

### 4. Learning System (`learning/`)
- `state_eval.py`: Board state evaluation
- `action_selection.py`: Move selection strategies
- `experience_processor.py`: Learning from gameplay
- Pattern recognition and value estimation

### 5. Agent Core (`agent/`)
- `core.py`: Main agent implementation
- Decision making process
- Strategy application
- Component integration

## Testing

Make sure you have installed the development dependencies (`pip install pytest pytest-cov`).

Run the full test suite:
```bash
python -m pytest -v
```

Run tests with coverage report:
```bash
python -m pytest --cov=tic_tac_toe.src --cov-report=html
```
This will generate an HTML coverage report in the `htmlcov/` directory.

## Development Status

**Current Version: 0.1.0** (as specified in `setup.py`)

**Status: Stable**

The core features are implemented and tested. The project includes:
- Complete game environment
- Functional agent with memory, reward, and learning systems
- Pattern recognition and strategic evaluation capabilities
- Self-play training mechanism
- Comprehensive test suite

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[License Type] - See LICENSE file for details 