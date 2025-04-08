import pytest
import numpy as np
from tic_tac_toe.src.environment import Board, GameEnvironment

def test_board_initialization():
    board = Board()
    assert board.board.shape == (3, 3)
    assert np.all(board.board == 0)
    assert board.current_player == 1
    assert len(board.move_history) == 0

def test_valid_moves():
    board = Board()
    assert len(board.get_valid_moves()) == 9  # All moves valid initially
    
    # Make a move
    assert board.make_move((0, 0))
    assert len(board.get_valid_moves()) == 8
    assert not board.is_valid_move((0, 0))  # Can't move in same spot

def test_win_conditions():
    board = Board()
    
    # Test row win
    board.make_move((0, 0))  # X
    board.make_move((1, 0))  # O
    board.make_move((0, 1))  # X
    board.make_move((1, 1))  # O
    board.make_move((0, 2))  # X
    assert board.check_winner() == 1  # X wins
    
    # Test column win
    board = Board()
    board.make_move((0, 0))  # X
    board.make_move((0, 1))  # O
    board.make_move((1, 0))  # X
    board.make_move((1, 1))  # O
    board.make_move((2, 0))  # X
    assert board.check_winner() == 1  # X wins
    
    # Test diagonal win
    board = Board()
    board.make_move((0, 0))  # X
    board.make_move((0, 1))  # O
    board.make_move((1, 1))  # X
    board.make_move((0, 2))  # O
    board.make_move((2, 2))  # X
    assert board.check_winner() == 1  # X wins

def test_draw_condition():
    board = Board()
    # Fill board without winner
    moves = [(0,0), (0,1), (0,2),
             (1,1), (1,0), (1,2),
             (2,1), (2,0), (2,2)]
    for move in moves:
        board.make_move(move)
    
    assert board.is_draw()
    assert board.check_winner() is None

def test_game_environment():
    env = GameEnvironment()
    
    # Test initial state
    state = env.reset()
    assert np.all(state == 0)
    assert len(env.get_valid_moves()) == 9
    
    # Test making a move
    state, reward, done, info = env.step((0, 0))
    assert state[0, 0] == 1
    assert not done
    assert reward == 0.0
    
    # Test invalid move
    state, reward, done, info = env.step((0, 0))
    assert reward == -10.0
    assert done
    assert 'error' in info
    
    # Test game completion
    env.reset()
    # Make moves for X to win
    moves = [(0,0), (1,0), (0,1), (1,1), (0,2)]
    for move in moves:
        state, reward, done, info = env.step(move)
    
    assert done
    assert info['winner'] == 1
    assert reward == -1.0  # O's perspective (last player to move)

def test_game_clone():
    env = GameEnvironment()
    
    # Make some moves
    env.step((0, 0))
    env.step((1, 1))
    
    # Clone the environment
    cloned_env = env.clone()
    
    # Check if states match
    assert np.array_equal(env.get_state(), cloned_env.get_state())
    assert env.get_current_player() == cloned_env.get_current_player()
    
    # Make sure they're independent
    env.step((0, 1))
    assert not np.array_equal(env.get_state(), cloned_env.get_state()) 