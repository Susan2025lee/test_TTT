import numpy as np
import pytest
from tic_tac_toe.src.agent.core import Agent
from tic_tac_toe.src.environment.board import Board
from tic_tac_toe.src.memory.memory_system import MemorySystem
from tic_tac_toe.src.reward.reward_system import RewardSystem
from tic_tac_toe.src.learning.state_eval import StateEvaluator
from tic_tac_toe.src.learning.action_selection import ActionSelector
from tic_tac_toe.src.learning.experience_processor import ExperienceProcessor

@pytest.fixture
def agent():
    """Create a test agent with all components."""
    memory_system = MemorySystem()
    reward_system = RewardSystem()
    state_evaluator = StateEvaluator()
    action_selector = ActionSelector(state_evaluator=state_evaluator)
    experience_processor = ExperienceProcessor(
        experience_memory=memory_system.experience,
        strategic_memory=memory_system.strategic,
        state_evaluator=state_evaluator
    )
    return Agent(
        memory_system=memory_system,
        reward_system=reward_system,
        state_evaluator=state_evaluator,
        action_selector=action_selector,
        experience_processor=experience_processor,
        player_id=1
    )

def test_move_selection(agent):
    """Test that agent can select valid moves."""
    board = Board()
    move = agent.select_move(board)
    assert move in board.get_valid_moves()
    assert len(move) == 2
    assert all(isinstance(x, int) for x in move)
    
def test_strategic_move_preference(agent):
    """Test that agent prefers strategic moves."""
    board = Board()
    # Make the center position highly valuable
    agent.memory_system.strategic.position_values[1, 1] = 0.9
    move = agent.select_move(board)
    assert move == (1, 1)  # Should prefer center
    
def test_winning_move_detection(agent):
    """Test that agent can detect and select winning moves via internal evaluation."""
    board = Board()
    # Set up a state where X (player 1) can win with (1, 2)
    # O X _ 
    # O X _ 
    # O _ X 
    board.board = np.array([
        [-1, 1, 0], 
        [-1, 1, 0], 
        [-1, 0, 1] 
    ])
    board.current_player = 1 # Ensure it's X's turn
    board.move_count = 6 # Update move count accordingly
    
    print("Board state for winning move test:")
    print(board.get_state())

    # Agent should evaluate moves and find the winning one (2, 1) in this setup
    # The _evaluate_moves method should assign float('inf') to (2, 1)
    # The ActionSelector should prioritize float('inf')
    move = agent.select_move(board)
    
    # Verify the agent selected the winning move
    expected_win_move = (2, 1)
    assert move == expected_win_move, f"Agent failed to select winning move {expected_win_move}, selected {move} instead."
    
def test_game_result_processing(agent):
    """Test that agent properly processes game results."""
    board = Board()
    # Play a short game
    moves = [(1, 1), (0, 0), (2, 2)]
    states = []
    
    for move in moves:
        states.append(board.get_state().copy())
        agent.memory_system.short_term.add_state(board)  # Pass the board object
        agent.memory_system.short_term.add_move(move)
        board.make_move(move)
    
    # Process win
    agent.update_from_game_result(1)
    
    # Check that experience was stored
    assert len(agent.memory_system.experience.experiences) == 1
    
def test_counter_move_selection(agent):
    """Test that agent can select good counter moves."""
    board = Board()
    # Set up a position where O just played center
    board.make_move((0, 0))  # X
    board.make_move((1, 1))  # O
    
    # Add a strong counter-move to strategic memory
    agent.memory_system.strategic.counter_moves[(1, 1)][(0, 2)] = {'wins': 8, 'total': 10}
    
    # Agent should prefer the counter-move
    move = agent.select_move(board)
    assert move == (0, 2)
    
def test_pattern_based_selection(agent):
    """Test that agent uses pattern recognition in move selection."""
    board = Board()
    # Create a diagonal pattern
    board.make_move((0, 0))  # X
    board.make_move((1, 0))  # O
    board.make_move((1, 1))  # X
    
    # Add value to diagonal pattern
    state = board.get_state()
    diag_pattern = f"diag_{''.join(map(str, np.diagonal(state).astype(int)))}"
    agent.memory_system.strategic.pattern_values[diag_pattern] = 0.8
    
    # Agent should prefer completing the diagonal
    move = agent.select_move(board)
    assert move == (2, 2) 