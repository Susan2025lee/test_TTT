import sys
from flask import Flask, render_template, request, jsonify

# Import necessary classes from your tic_tac_toe package
# Assuming your package is installed and accessible
from tic_tac_toe.src.environment.game_env import GameEnvironment
from tic_tac_toe.src.agent.core import Agent
from tic_tac_toe.src.memory.memory_system import MemorySystem
from tic_tac_toe.src.reward.reward_system import RewardSystem
from tic_tac_toe.src.learning.state_eval import StateEvaluator
from tic_tac_toe.src.learning.action_selection import ActionSelector
from tic_tac_toe.src.learning.experience_processor import ExperienceProcessor

app = Flask(__name__)

# --- Game Initialization ---
# Global variables to hold the game state and agent
# WARNING: Using global variables is simple for this example, but not ideal for
# production apps (concurrency issues). Consider session management or other
# state management techniques for more complex scenarios.
game_env = None
agent = None
human_player_id = 1
agent_player_id = -1

def create_agent_instance(player_id):
    """Initializes all components and creates an agent instance."""
    memory = MemorySystem()
    reward = RewardSystem()
    evaluator = StateEvaluator()
    selector = ActionSelector(state_evaluator=evaluator) # Agent plays deterministically
    processor = ExperienceProcessor(memory.experience, memory.strategic, evaluator)
    agent_instance = Agent(player_id=player_id,
                         memory_system=memory,
                         reward_system=reward,
                         state_evaluator=evaluator,
                         action_selector=selector,
                         experience_processor=processor)
    # Add loading logic here if you have saved agents
    return agent_instance

def initialize_game():
    """Sets up a new game environment and agent."""
    global game_env, agent
    game_env = GameEnvironment()
    # Create agent instance (ensure this function is defined as before)
    agent = create_agent_instance(agent_player_id)
    print("New game initialized.")

# Initialize the first game when the app starts
initialize_game()

# --- Routes --- 

@app.route('/')
def index():
    """Serve the main game page."""
    return render_template('index.html')

@app.route('/get_state', methods=['GET'])
def get_state():
    """Return the current state of the game."""
    if not game_env:
        return jsonify({'error': 'Game not initialized'}), 500
    
    board_state = game_env.get_board().get_state().tolist() # Convert numpy array to list for JSON
    current_player = game_env.get_current_player()
    is_over = game_env.is_game_over()
    winner_val = game_env.get_winner() 
    # Explicitly convert NumPy int types to Python int for JSON serialization
    winner = int(winner_val) if winner_val is not None else None 
    
    return jsonify({
        'board': board_state,
        'currentPlayer': current_player,
        'isGameOver': is_over,
        'winner': winner, # Use the converted value
        'humanPlayerId': human_player_id
    })

@app.route('/move', methods=['POST'])
def handle_move():
    """Handle a move from the player or trigger the agent's move."""
    global game_env, agent

    if not game_env or not agent:
        return jsonify({'error': 'Game not initialized'}), 500

    if game_env.is_game_over():
        return jsonify({'error': 'Game is already over'}), 400

    current_player = game_env.get_current_player()
    agent_move = None

    # --- Human Move --- 
    if current_player == human_player_id:
        data = request.get_json()
        if not data or 'move' not in data:
            return jsonify({'error': 'Move data missing'}), 400
        
        try:
            move = tuple(data['move']) # Expecting [row, col]
            if len(move) != 2 or not all(isinstance(i, int) for i in move):
                 raise ValueError("Invalid move format")
        except (TypeError, ValueError):
             return jsonify({'error': 'Invalid move format, expected [row, col]'}), 400
        
        print(f"Received human move: {move}")
        if game_env.is_valid_move(move):
            game_env.make_move(move)
            print(f"Human move {move} applied.")
        else:
            print(f"Invalid human move attempt: {move}")
            # Return current state without making the invalid move
            return get_state()

    # --- Agent Move (Triggered after human move if game not over) --- 
    if not game_env.is_game_over() and game_env.get_current_player() == agent_player_id:
        print("Agent's turn...")
        board = game_env.get_board()
        agent_move = agent.select_move(board)
        if agent_move and game_env.is_valid_move(agent_move):
            game_env.make_move(agent_move)
            print(f"Agent move {agent_move} applied.")
        else:
            # Handle case where agent fails to move (should be rare)
            print(f"Agent failed to make a valid move (selected: {agent_move})")
            # Potentially just return state, or error
            # For now, we just proceed, but this indicates an agent logic issue
            pass 

    # Return the updated game state after moves
    return get_state()

@app.route('/reset', methods=['POST'])
def reset_game():
    """Reset the game to the initial state."""
    initialize_game()
    return get_state()

# --- Run the App --- 
if __name__ == '__main__':
    # Note: debug=True is helpful for development but should be False in production
    app.run(debug=True) 