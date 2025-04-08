import sys
# Import necessary classes from the installed package
from tic_tac_toe.src.environment.game_env import GameEnvironment
from tic_tac_toe.src.agent.core import Agent
from tic_tac_toe.src.memory.memory_system import MemorySystem
from tic_tac_toe.src.reward.reward_system import RewardSystem
from tic_tac_toe.src.learning.state_eval import StateEvaluator
from tic_tac_toe.src.learning.action_selection import ActionSelector
from tic_tac_toe.src.learning.experience_processor import ExperienceProcessor

# Helper function to create a standard agent (from user_guide.md)
def create_agent(player_id=1):
    """Initializes all components and creates an agent."""
    memory = MemorySystem()
    reward = RewardSystem()
    evaluator = StateEvaluator()
    selector = ActionSelector(state_evaluator=evaluator) # Uses default epsilon-greedy
    processor = ExperienceProcessor(memory.experience, memory.strategic, evaluator)
    agent = Agent(player_id=player_id,
                  memory_system=memory,
                  reward_system=reward,
                  state_evaluator=evaluator,
                  action_selector=selector,
                  experience_processor=processor)
    # Optional: Load pre-trained state if available (see user guide on saving/loading)
    # try:
    #     with open('my_trained_agent.pkl', 'rb') as f:
    #         loaded_agent_state = pickle.load(f) # Or load specific parts like memory
    #         # Apply loaded state to agent components here...
    # except FileNotFoundError:
    #     print("No pre-trained agent found, starting fresh.")
    # except Exception as e:
    #     print(f"Error loading agent state: {e}")

    return agent

# --- Game Setup ---
env = GameEnvironment() # Manages the game board and rules
agent = create_agent(player_id=-1) # Agent plays as O (-1)
human_player_id = 1 # You play as X (1)

print("-------------------------------------")
print(" Tic Tac Toe: Human (X) vs Agent (O)")
print("-------------------------------------")
print("Enter moves as row and column numbers (0, 1, or 2).")

# --- Game Loop ---
while not env.is_game_over():
    current_player = env.get_current_player()
    board = env.get_board() # Get the current board object
    print("\nCurrent board:")
    print(board) # Uses the Board's __str__ method

    if current_player == human_player_id:
        # --- Human player's turn ---
        valid_moves = env.get_valid_moves()
        print(f"\nYour turn (X). Valid moves: {valid_moves}")
        chosen_move = None
        while chosen_move is None:
            try:
                row_str = input("Enter row (0-2): ").strip()
                if not row_str: continue

                col_str = input("Enter column (0-2): ").strip()
                if not col_str: continue

                row = int(row_str)
                col = int(col_str)
                potential_move = (row, col)

                if env.is_valid_move(potential_move):
                    chosen_move = potential_move
                else:
                    print("Invalid move (check if out of bounds or position taken). Try again.")
            except ValueError:
                print("Invalid input. Please enter numbers (0, 1, or 2).")
            except KeyboardInterrupt:
                print("\nGame aborted by user.")
                sys.exit()

    else: # --- Agent's turn ---
        print(f"\nAgent's turn (O)... Thinking...")
        # Agent selects a move based on its logic
        chosen_move = agent.select_move(board)
        if chosen_move:
            print(f"Agent plays: {chosen_move}")
        else:
             # Should only happen if no valid moves are left, but game isn't over? (Error state)
             print("Error: Agent failed to select a move despite available moves. Exiting.")
             break

    # --- Apply the chosen move ---
    if chosen_move:
        env.make_move(chosen_move)
    else:
        # If no move was chosen (e.g., agent error), break loop
        break

# --- Game Over ---
print("\n---------- GAME OVER ----------")
print("Final board:")
print(env.get_board())

winner = env.get_winner() # Check the winner from the environment
if winner == human_player_id:
    print("Congratulations! You (X) won!")
elif winner == agent.player_id:
    print("The Agent (O) won!")
else:
    print("It's a draw!")
print("-----------------------------")

# Note: The agent doesn't learn from this single game unless you explicitly
# call agent.update_from_game_result(outcome) here, which isn't typical
# for just playing. Learning usually happens during dedicated training phases.