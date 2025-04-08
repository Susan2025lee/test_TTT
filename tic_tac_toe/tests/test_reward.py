import pytest
import numpy as np
from tic_tac_toe.src.environment import Board
from tic_tac_toe.src.reward import RewardSystem

def test_reward_initialization():
    reward_system = RewardSystem()
    
    # Check default reward values
    assert reward_system.win_reward == 1.0
    assert reward_system.loss_reward == -1.0
    assert reward_system.draw_reward == 0.0
    assert reward_system.invalid_move_reward == -10.0
    
    # Check position values
    assert reward_system.position_values[1, 1] == 0.4  # Center
    assert reward_system.position_values[0, 0] == 0.3  # Corner
    assert reward_system.position_values[0, 1] == 0.2  # Edge

def test_invalid_move_reward():
    reward_system = RewardSystem()
    board = Board()
    
    # Make a move
    board.make_move((0, 0))
    
    # Try to make same move again
    reward, breakdown = reward_system.calculate_reward(board, (0, 0))
    
    assert reward == reward_system.invalid_move_reward
    assert 'invalid_move' in breakdown
    assert breakdown['invalid_move'] == reward_system.invalid_move_reward

def test_terminal_rewards():
    reward_system = RewardSystem()
    board = Board()
    
    # Create a winning state for X
    moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
    for move in moves:
        board.make_move(move)
    
    # Test reward for O (loser)
    reward, breakdown = reward_system.calculate_reward(board, (1, 2), is_terminal=True)
    assert reward == reward_system.loss_reward
    assert breakdown['terminal'] == reward_system.loss_reward
    
    # Reset and create a draw
    board = Board()
    moves = [(0, 0), (1, 1), (0, 1), (0, 2), (2, 0), (1, 0), (1, 2), (2, 1), (2, 2)]
    for move in moves:
        board.make_move(move)
    
    reward, breakdown = reward_system.calculate_reward(board, (2, 2), is_terminal=True)
    assert reward == reward_system.draw_reward
    assert breakdown['terminal'] == reward_system.draw_reward

def test_position_rewards():
    reward_system = RewardSystem()
    board = Board()
    
    # Test center move
    reward, breakdown = reward_system.calculate_reward(board, (1, 1))
    assert breakdown['position'] == 0.4
    
    # Test corner move
    reward, breakdown = reward_system.calculate_reward(board, (0, 0))
    assert breakdown['position'] == 0.3
    
    # Test edge move
    reward, breakdown = reward_system.calculate_reward(board, (0, 1))
    assert breakdown['position'] == 0.2

def test_tactical_rewards():
    reward_system = RewardSystem()
    board = Board()
    
    # Set up a state where next move can block opponent's win
    board.make_move((0, 0))  # X
    board.make_move((1, 0))  # O
    board.make_move((0, 1))  # X
    board.make_move((1, 1))  # O
    
    # Test blocking move
    reward, breakdown = reward_system.calculate_reward(board, (1, 2))
    assert breakdown['tactical'] >= reward_system.blocking_reward
    
    # Test winning opportunity
    board = Board()
    board.make_move((0, 0))  # X
    board.make_move((1, 1))  # O
    board.make_move((0, 1))  # X
    
    reward, breakdown = reward_system.calculate_reward(board, (0, 2))
    assert breakdown['tactical'] >= reward_system.winning_opportunity_reward

def test_progress_rewards():
    reward_system = RewardSystem()
    board = Board()
    
    # Test early game center move
    reward, breakdown = reward_system.calculate_reward(board, (1, 1))
    assert breakdown['progress'] > 0
    
    # Test early game corner move
    reward, breakdown = reward_system.calculate_reward(board, (0, 0))
    assert breakdown['progress'] > 0
    
    # Set up mid-game state
    moves = [(0, 0), (1, 1), (2, 2), (1, 0)]
    for move in moves:
        board.make_move(move)
    
    # Test mid-game move
    reward, breakdown = reward_system.calculate_reward(board, (0, 1))
    assert 'progress' in breakdown

def test_reward_updates():
    reward_system = RewardSystem()
    initial_blocking = reward_system.blocking_reward
    initial_winning = reward_system.winning_opportunity_reward
    
    # Test win adjustment
    reward_system.update_rewards(1)
    after_win_blocking = reward_system.blocking_reward
    after_win_winning = reward_system.winning_opportunity_reward
    assert after_win_blocking < initial_blocking
    assert after_win_winning > initial_winning
    
    # Test loss adjustment
    reward_system.update_rewards(-1)
    after_loss_blocking = reward_system.blocking_reward
    after_loss_winning = reward_system.winning_opportunity_reward
    assert after_loss_blocking > after_win_blocking
    assert after_loss_winning < after_win_winning
    
    # Test bounds
    for _ in range(10):
        reward_system.update_rewards(1)
    
    assert 0.1 <= reward_system.blocking_reward <= 0.5
    assert 0.1 <= reward_system.winning_opportunity_reward <= 0.5
    assert 0.2 <= reward_system.fork_creation_reward <= 0.6 