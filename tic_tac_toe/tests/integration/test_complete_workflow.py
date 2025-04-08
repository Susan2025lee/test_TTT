"""Integration tests for the complete Tic Tac Toe agent system workflow.

This test suite verifies that all components of the system work together correctly,
including training, gameplay, memory systems, and performance tracking.
"""

import pytest
import numpy as np
from tic_tac_toe.src.agent.core import Agent
from tic_tac_toe.src.environment.game_env import GameEnvironment
from tic_tac_toe.src.training.self_play import SelfPlayTrainer
from tic_tac_toe.src.memory.memory_system import MemorySystem
from tic_tac_toe.src.reward.reward_system import RewardSystem
from tic_tac_toe.src.learning.state_eval import StateEvaluator
from tic_tac_toe.src.learning.action_selection import ActionSelector
from tic_tac_toe.src.learning.experience_processor import ExperienceProcessor
import time
import cProfile
import pstats
import io
import os

def create_agent(player_id: int = 1) -> Agent:
    """Helper function to create an agent with all required components."""
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
        player_id=player_id,
        memory_system=memory_system,
        reward_system=reward_system,
        state_evaluator=state_evaluator,
        action_selector=action_selector,
        experience_processor=experience_processor
    )

def test_agent_training_and_gameplay():
    """Test complete workflow of agent training and gameplay."""
    # Create agent with custom strategic patterns
    agent = create_agent()
    agent.memory_system.strategic.update_pattern_value("row_110", 0.8)
    agent.memory_system.strategic.update_pattern_value("diag_110", 0.9)
    
    # Create trainer and run training
    trainer = SelfPlayTrainer(agent, num_games=100, eval_frequency=20)
    metrics = trainer.train()
    
    # Verify training metrics
    assert len(metrics['win_rates']) > 0
    assert len(metrics['draw_rates']) > 0
    assert len(metrics['avg_game_length']) > 0
    assert len(metrics['learning_curves']) > 0
    
    # Set up a position where O must block X's immediate win threat
    env = GameEnvironment()
    env.reset()
    
    # Set up the board state
    board = env.get_board()
    env.make_move((0, 0))  # X plays top-left
    env.make_move((1, 1))  # O plays center
    env.make_move((0, 1))  # X plays top-middle, creating immediate win threat
    
    # Agent (O) must detect and block the immediate win threat
    move = agent.select_move(env.get_board())
    env.make_move(move)
    
    # Verify agent chose the only blocking move that prevents immediate loss
    assert move == (0, 2), f"Agent failed to block X's immediate win threat at (0,2), played {move} instead"
    # Verify game continues (not over) since this was a blocking move
    assert not env.is_game_over(), "Game should not be over after blocking move"
    # Verify no winner yet
    assert env.get_winner() is None, "There should not be a winner yet"

def test_agent_vs_agent_gameplay():
    """Test gameplay between two trained agents."""
    # Create and train two agents
    agent1 = create_agent(player_id=1)
    agent2 = create_agent(player_id=-1)
    
    # Train both agents
    trainer = SelfPlayTrainer(agent1, agent2, num_games=50, eval_frequency=25)
    metrics = trainer.train()
    
    # Play a series of games between agents
    env = GameEnvironment()
    wins = {1: 0, -1: 0, 0: 0}  # Track wins for each player and draws
    
    for _ in range(10):
        env.reset()
        while not env.is_game_over():
            current_agent = agent1 if env.get_current_player() == 1 else agent2
            move = current_agent.select_move(env.get_board())
            env.make_move(move)
        
        winner = env.get_winner()
        wins[0 if winner is None else winner] += 1
    
    # Verify reasonable distribution of outcomes
    total_games = sum(wins.values())
    assert total_games == 10
    assert wins[0] > 0, "No draws occurred, which is unlikely in optimal play"

@pytest.mark.skip(reason="Test assumes memory updates happen mid-game, but they occur post-game.")
def test_memory_integration():
    """Test integration of memory systems during gameplay."""
    agent = create_agent()
    env = GameEnvironment()
    
    # Play a game and verify memory updates
    env.reset()
    initial_state = env.get_board()
    
    # Make some moves
    moves = [(1, 1), (0, 0), (2, 2)]
    for move in moves:
        agent.select_move(env.get_board())  # This should update short-term memory
        env.make_move(move)
    
    # Verify short-term memory
    last_move = agent.memory_system.short_term.get_last_move()
    assert last_move is not None, "Short-term memory failed to record moves"
    
    # Verify strategic memory updates
    assert len(agent.memory_system.strategic.pattern_values) > 0, "Strategic memory not updating"

def test_performance_under_load():
    """Test system performance under higher load and profile training."""
    agent = create_agent()
    # Reduce games for faster profiling, increase later if needed
    num_games_for_profile = 50 
    trainer = SelfPlayTrainer(agent, num_games=num_games_for_profile, eval_frequency=25)
    
    profiler = cProfile.Profile()
    print(f"\nStarting profiling for {num_games_for_profile} games...")
    profiler.enable()
    
    # Time the training process
    start_time = time.time()
    metrics = trainer.train()
    end_time = time.time()
    
    profiler.disable()
    print("Profiling complete.")
    
    training_time = end_time - start_time
    print(f"Training {num_games_for_profile} games took: {training_time:.2f} seconds")

    # Save profiling stats to a file
    stats_file_path = "profile_stats.txt" # Save in workspace root
    try:
        with open(stats_file_path, 'w') as f:
            sortby = pstats.SortKey.CUMULATIVE # Sort by cumulative time spent in function
            ps = pstats.Stats(profiler, stream=f).sort_stats(sortby)
            ps.print_stats(30) # Save top 30 functions
        print(f"Profiling results saved to: {os.path.abspath(stats_file_path)}")
    except Exception as e:
        print(f"Error saving profiling stats: {e}")

    # Basic assertions (adjust thresholds based on typical performance)
    assert training_time < 30, f"Training {num_games_for_profile} games took too long: {training_time:.2f} seconds"
    assert len(metrics['win_rates']) > 0
    assert len(metrics['draw_rates']) > 0
    assert len(metrics['learning_curves']) > 0

def test_error_handling():
    """Test system error handling and recovery."""
    agent = create_agent()
    env = GameEnvironment()
    
    # Test invalid move handling
    env.reset()
    env.make_move((1, 1))  # Make center position occupied
    
    # Try to make invalid move
    result = env.make_move((1, 1))
    assert not result, "Invalid move was accepted"
    
    # Verify agent handles invalid move gracefully
    move = agent.select_move(env.get_board())
    assert move != (1, 1), "Agent selected invalid move"
    assert env.is_valid_move(move), "Agent selected invalid move"

def test_reward_system_integration():
    """Test integration of reward system with agent learning."""
    # Ensure agent uses player_id 1 for this test
    agent = create_agent(player_id=1) 
    env = GameEnvironment()
    # Ensure short-term memory is clear before the test game
    agent.reset() 

    # Play a game with specific moves that *guarantee* a win for player 1 (agent)
    env.reset()
    # Winning sequence for player 1 (X): Center, Top-Mid, Bottom-Mid (via Middle Column)
    moves = [(1, 1), (0, 0), (0, 1), (2, 0), (2, 1)] 
    states_history = [env.get_board().copy()] # Store initial state

    print("Simulating game for learning (Player 1 WIN)...")
    for i, move in enumerate(moves):
        current_state = states_history[-1]
        current_player = env.get_current_player()
        print(f"Turn {i}: Player {current_player} considers move {move} from state:\n{current_state}")
        
        # Store state BEFORE the move and the move itself in agent's short-term memory
        agent.memory_system.short_term.add_state(current_state)
        agent.memory_system.short_term.add_move(move)
        
        # Execute move in environment
        env.make_move(move)
        # Store state AFTER the move for the next iteration
        states_history.append(env.get_board().copy())
        print(f"Board after move {move}:\n{states_history[-1]}")

        if env.is_game_over():
            print(f"Game over detected after move {i+1}.")
            break # Exit loop once game ends

    # Game finished, update agent based on outcome
    winner = env.get_winner()
    print(f"Game finished. Winner: {winner}")
    outcome = 0
    if winner == agent.player_id:
        outcome = 1
        print("Agent (Player 1) won as expected.")
    elif winner == -agent.player_id:
        # This case should not happen with the chosen move sequence
        outcome = -1
        print("ERROR: Agent lost unexpectedly.")
    else:
        # This case should not happen with the chosen move sequence
        print("ERROR: Game ended in a draw unexpectedly.")

    # Call the update method - this uses the history stored in short_term memory
    assert outcome == 1, "Test setup error: Simulated game did not result in a win for Player 1."
    print(f"Updating agent from game result (Outcome: {outcome})...")
    agent.update_from_game_result(outcome)
    print("Agent update complete.")

    # --- Verification Phase --- 
    print("\nStarting verification phase...")
    env.reset()
    # Set up the board state *exactly* as it was before the agent's winning move (2,1)
    # in the learning game. Sequence: (1,1)X -> (0,0)O -> (0,1)X -> (2,0)O
    verification_moves = [(1, 1), (0, 0), (0, 1), (2, 0)]
    for move in verification_moves:
        env.make_move(move)
    
    print("Board state for verification decision (X to move):")
    board_state_for_decision = env.get_board()
    print(board_state_for_decision)
    assert board_state_for_decision.current_player == agent.player_id, f"Verification state has wrong player: {board_state_for_decision.current_player}"

    # Agent (Player 1, X) should now recognize the winning move (2,1)
    print("Agent selecting move...")
    move = agent.select_move(board_state_for_decision)
    print(f"Agent selected move: {move}")

    # Expected winning move for X based on the learned middle column win
    expected_move = (2, 1)
    assert move == expected_move, (
        f"Agent failed to select the learned winning move. \n"
        f"Selected: {move}, Expected: {expected_move}\n"
        f"Pattern values sample: { {k: v for k, v in agent.memory_system.strategic.pattern_values.items() if '111' in k or '110' in k or '011' in k or '101' in k} }"
    )
    print("Verification successful: Agent selected an expected winning move.") 