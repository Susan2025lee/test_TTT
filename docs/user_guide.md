# Tic Tac Toe Agent User Guide

## Introduction

Welcome to the user guide for the Tic Tac Toe AI agent system. This guide provides instructions for installing, using, and understanding the components of this project. Whether you want to play against the agent, train it further, or learn about its architecture, this guide is your starting point.

## Installation

Follow these steps to install the Tic Tac Toe agent system:

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
    Using a virtual environment is highly recommended to avoid conflicts with other Python projects.

3.  **Install the package:**
    Navigate to the root directory of the cloned repository (where `setup.py` is located) and run:

    *   **For regular use:**
        ```bash
        pip install .
        ```
    *   **For development (if you plan to modify the code):**
        ```bash
        pip install -e .
        ```
    This command installs the `tic_tac_toe_agent` package and its core dependency (NumPy).

4.  **(Optional) Install development dependencies:**
    If you intend to run tests or use development tools, install the extra dependencies:
    ```bash
    pip install pytest pytest-cov black flake8
    ```

## Quick Start

This section provides examples for quickly playing against the agent or training it.

### Option 1: Playing via Command-Line Interface (CLI)

This example shows how to set up a simple text-based game in your terminal where you play against the AI agent.

1.  Ensure the package is installed (see Installation).
2.  Create a Python script (e.g., `play_cli.py`) and paste the following code:

```python
# --- Paste the CLI game loop code here ---
# (Refer to the previous version of this guide or README for the full script)
import sys
from tic_tac_toe.src.environment.game_env import GameEnvironment
from tic_tac_toe.src.agent.core import Agent
from tic_tac_toe.src.memory.memory_system import MemorySystem
from tic_tac_toe.src.reward.reward_system import RewardSystem
from tic_tac_toe.src.learning.state_eval import StateEvaluator
from tic_tac_toe.src.learning.action_selection import ActionSelector
from tic_tac_toe.src.learning.experience_processor import ExperienceProcessor

def create_agent(player_id=1):
    # ... (rest of create_agent function as defined previously)
    memory = MemorySystem()
    reward = RewardSystem()
    evaluator = StateEvaluator()
    selector = ActionSelector(state_evaluator=evaluator)
    processor = ExperienceProcessor(memory.experience, memory.strategic, evaluator)
    agent = Agent(player_id=player_id,
                  memory_system=memory,
                  reward_system=reward,
                  state_evaluator=evaluator,
                  action_selector=selector,
                  experience_processor=processor)
    return agent

env = GameEnvironment()
agent = create_agent(player_id=-1)
human_player_id = 1

print("Starting CLI game: Human (X) vs Agent (O)")
while not env.is_game_over():
    current_player = env.get_current_player()
    board = env.get_board()
    print("\nCurrent board:")
    print(board)
    if current_player == human_player_id:
        valid_moves = env.get_valid_moves()
        print(f"\nYour turn (X). Valid moves: {valid_moves}")
        chosen_move = None
        while chosen_move is None:
            try:
                row_str = input("Enter row (0-2): ").strip()
                col_str = input("Enter column (0-2): ").strip()
                if row_str and col_str:
                    row = int(row_str)
                    col = int(col_str)
                    potential_move = (row, col)
                    if env.is_valid_move(potential_move):
                        chosen_move = potential_move
                    else:
                        print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Please enter numbers.")
            except KeyboardInterrupt:
                print("\nGame aborted.")
                sys.exit()
    else:
        print(f"\nAgent's turn (O)...")
        chosen_move = agent.select_move(board)
        if chosen_move:
            print(f"Agent plays: {chosen_move}")
        else:
            break # Agent error
    if chosen_move:
        env.make_move(chosen_move)
    else:
        break # Error

print("\n----- Game Over -----")
print("Final board:")
print(env.get_board())
winner = env.get_winner()
# ... (print winner message as before) ...
if winner == human_player_id: print("You won!")
elif winner == agent.player_id: print("Agent won!")
else: print("Draw!")
```
3.  Run the script from your terminal:
    ```bash
    python play_cli.py
    ```

### Option 2: Playing via Web Interface

This project also includes a graphical web interface using Flask.

1.  Ensure the package and its dependencies (including Flask) are installed:
    ```bash
    # In project root with venv active:
    pip install -e .
    pip install -r requirements.txt
    ```
2.  Run the Flask application from the project root directory:
    ```bash
    python app.py
    ```
3.  Open your web browser and go to `http://127.0.0.1:5000` (or the address shown in the terminal).
4.  Click on the board cells to make your move. Use the "New Game" button to reset.

### Training Your Own Agent

This example demonstrates how to train an agent using the self-play mechanism.

```python
import matplotlib.pyplot as plt
from tic_tac_toe.src.training.self_play import SelfPlayTrainer
from tic_tac_toe.src.agent.core import Agent
from tic_tac_toe.src.memory.memory_system import MemorySystem
from tic_tac_toe.src.reward.reward_system import RewardSystem
from tic_tac_toe.src.learning.state_eval import StateEvaluator
from tic_tac_toe.src.learning.action_selection import ActionSelector
from tic_tac_toe.src.learning.experience_processor import ExperienceProcessor

# Helper function (same as above)
def create_agent(player_id=1):
    memory = MemorySystem()
    reward = RewardSystem()
    evaluator = StateEvaluator()
    selector = ActionSelector(exploration_rate=0.1) # Example: Set exploration rate
    processor = ExperienceProcessor(memory.experience, memory.strategic, evaluator)
    agent = Agent(player_id=player_id,
                  memory_system=memory,
                  reward_system=reward,
                  state_evaluator=evaluator,
                  action_selector=selector,
                  experience_processor=processor)
    return agent

# --- Training Setup ---
agent_to_train = create_agent(player_id=1)
trainer = SelfPlayTrainer(
    agent=agent_to_train,
    num_games=2000, # Number of self-play games
    eval_frequency=200 # How often to record metrics (optional)
)

# --- Train Agent ---
print(f"Starting training for {trainer.num_games} games...")
metrics = trainer.train()
print("\nTraining completed!")

# --- Display Metrics (Optional) ---
if metrics and metrics.get('total_games_played', 0) > 0:
    print(f"Trained over {metrics['total_games_played']} games.")
    print(f"Final Win Rate (vs self): {metrics.get('win_rate', 'N/A'):.2%}")
    print(f"Final Draw Rate (vs self): {metrics.get('draw_rate', 'N/A'):.2%}")
    print(f"Final Loss Rate (vs self): {metrics.get('loss_rate', 'N/A'):.2%}")

    # Example of plotting (requires matplotlib)
    # Check if metrics needed for plotting exist
    if 'game_numbers' in metrics and 'win_rates' in metrics:
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(metrics['game_numbers'], metrics['win_rates'], label='Win Rate')
            plt.plot(metrics['game_numbers'], metrics['draw_rates'], label='Draw Rate')
            plt.plot(metrics['game_numbers'], metrics['loss_rates'], label='Loss Rate')
            plt.title('Agent Performance During Self-Play Training')
            plt.xlabel('Number of Games Played')
            plt.ylabel('Rate')
            plt.legend()
            plt.grid(True)
            plt.show()
        except ImportError:
            print("\nMatplotlib not installed. Skipping plot.")
            print("Install with: pip install matplotlib")
else:
    print("No metrics recorded or training did not run.")

# You can now potentially save or use the 'agent_to_train'
# For saving, see 'Advanced Usage' below.
```

## System Components

For a deeper understanding of how the agent works, refer to the detailed documentation for each major component:

*   **[Game Environment](environment.md)**: Defines the rules and state of the Tic Tac Toe game.
*   **[Memory System](memory_system.md)**: Manages how the agent stores and recalls information (short-term, experience, strategic).
*   **[Reward System](reward_system.md)**: Calculates rewards based on game events and outcomes.
*   **[Learning System](learning_system.md)**: Handles state evaluation, action selection, and learning from experience.
*   **[Agent Core](agentcore_system.md)**: The central agent class that integrates all other components.
*   **[Training System](training_system.md)**: Orchestrates the self-play training process.

## Advanced Usage

### Customizing Agent Components

You can create instances of components like `ActionSelector` or `StateEvaluator` with custom parameters and pass them during agent creation.

```python
# Example: Create agent with a different exploration rate
custom_selector = ActionSelector(exploration_rate=0.05)

agent = Agent(
    player_id=1,
    memory_system=MemorySystem(), # Use default memory
    reward_system=RewardSystem(), # Use default reward
    state_evaluator=StateEvaluator(), # Use default evaluator
    action_selector=custom_selector, # Use custom selector
    experience_processor=ExperienceProcessor(...) # Requires other components
)
```
*(Note: Ensure all dependencies are correctly provided when customizing components.)*

### Saving and Loading Agents

Saving the state of a trained agent allows you to reuse it later without retraining. Python's `pickle` is a straightforward way to do this for simple objects, but be aware of potential versioning issues if the class structure changes.

```python
import pickle

# Assume 'trained_agent' is your agent object after training
agent_filename = 'my_trained_agent.pkl'

# Save agent
try:
    with open(agent_filename, 'wb') as f:
        pickle.dump(trained_agent, f)
    print(f"Agent saved to {agent_filename}")
except Exception as e:
    print(f"Error saving agent: {e}")

# Load agent later
try:
    with open(agent_filename, 'rb') as f:
        loaded_agent = pickle.load(f)
    print(f"Agent loaded from {agent_filename}")
    # Now you can use loaded_agent
except FileNotFoundError:
    print(f"Error: Agent file '{agent_filename}' not found.")
except Exception as e:
    print(f"Error loading agent: {e}")
```
*Recommendation: For more robust serialization, consider libraries like `joblib` or custom saving methods that store parameters rather than entire objects.*

## Troubleshooting

*   **Installation Issues**: Ensure you are using Python 3.7+ and have `pip` available. Check permissions if installation fails. Use a virtual environment to isolate dependencies.
*   **Import Errors**: Verify that you installed the package correctly (`pip install .` or `pip install -e .`). Imports should generally start from `tic_tac_toe.src...` if you are running scripts from outside the `src` directory after installation.
*   **Agent Not Learning**: 
    *   Increase the number of training games (`num_games`).
    *   Adjust the exploration rate (`epsilon` in `ActionSelector`). Too low might prevent discovery; too high might prevent convergence.
    *   Verify the `RewardSystem` provides meaningful feedback.
    *   Check the implementation of the `ExperienceProcessor` and memory updates.
*   **Invalid Moves during Gameplay**: Ensure your input logic correctly validates moves against `env.is_valid_move()` before applying them.

## Contributing

Contributions are welcome! Please follow standard practices:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/YourFeature`).
3.  Make your changes and commit them (`git commit -m 'Add YourFeature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

Please ensure your code adheres to formatting standards (e.g., using Black) and includes tests where applicable.

## Support

If you encounter issues or have questions:

1.  Consult this User Guide and the specific component documentation.
2.  Check the `README.md` file.
3.  Search existing issues on the project's repository.
4.  If your issue is new, create a detailed report including:
    *   A clear description of the problem.
    *   Steps to reproduce the issue.
    *   Expected behavior vs. actual behavior.
    *   Relevant error messages or logs. 