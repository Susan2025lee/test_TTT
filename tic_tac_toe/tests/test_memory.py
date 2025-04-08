import pytest
import numpy as np
from tic_tac_toe.src.environment import Board
from tic_tac_toe.src.memory import (
    ShortTermMemory,
    ExperienceMemory,
    StrategicMemory,
    MemorySystem
)

def test_short_term_memory():
    memory = ShortTermMemory()
    board = Board()
    
    # Test adding states
    memory.add_state(board)
    assert len(memory.board_states) == 1
    
    # Test adding moves
    memory.add_state(board, (0, 0), 0.5)
    assert len(memory.current_sequence) == 1
    assert memory.get_move_score((0, 0)) == 0.5
    
    # Test history limit
    memory = ShortTermMemory(max_history_length=2)
    for _ in range(3):
        memory.add_state(board)
    assert len(memory.board_states) == 2

def test_experience_memory():
    memory = ExperienceMemory()
    
    # Create sample game
    states = [np.zeros((3, 3)) for _ in range(3)]
    moves = [(0, 0), (1, 1), (0, 1)]
    
    # Test adding experience
    memory.add_experience(states, moves, 1, [1, -1, 1])
    
    # Test pattern retrieval
    similar = memory.get_similar_states(states[0])
    assert len(similar) > 0
    
    # Test move statistics
    stats = memory.get_move_stats((0, 0))
    assert stats['total'] == 1
    assert stats['wins'] == 1

def test_strategic_memory():
    memory = StrategicMemory()
    
    # Test initial position values
    assert memory.get_position_value((1, 1)) == 0.4  # Center value
    assert memory.get_position_value((0, 0)) == 0.3  # Corner value
    
    # Test adding game
    states = [np.zeros((3, 3)) for _ in range(3)]
    moves = [(0, 0), (1, 1), (0, 1)]
    memory.add_game(states, moves, 1)
    
    # Test opening moves
    best_opening = memory.get_best_opening()
    assert best_opening is not None
    
    # Test counter moves
    counters = memory.get_best_counter((0, 0))
    assert len(counters) > 0

def test_memory_system():
    system = MemorySystem()
    board = Board()
    
    # Test adding state
    system.add_state(board)
    assert len(system.short_term.board_states) == 1
    
    # Make some moves
    board.make_move((0, 0))  # X
    system.add_state(board, (0, 0))
    board.make_move((1, 1))  # O
    system.add_state(board, (1, 1))
    board.make_move((0, 1))  # X
    system.add_state(board, (0, 1))
    
    # Test move recommendations
    recommendations = system.get_move_recommendation(board)
    assert len(recommendations) > 0
    
    # Test end game processing
    system.end_game(1)  # X wins
    
    # Check that short-term memory was reset
    assert len(system.short_term.board_states) == 0
    
    # Check that experience was stored
    assert len(system.experience.experiences) == 1
    
    # Test context retrieval
    context = system.get_context()
    assert 'short_term' in context
    assert 'best_moves' in context
    assert 'position_values' in context

def test_memory_system_integration():
    system = MemorySystem()
    board = Board()
    
    # Play multiple games
    for _ in range(3):
        board.reset()
        system.add_state(board)
        
        while not board.is_game_over():
            # Get move recommendation
            recommendations = system.get_move_recommendation(board)
            if recommendations:
                move = recommendations[0][0]
            else:
                move = board.get_valid_moves()[0]
                
            # Make move
            board.make_move(move)
            system.add_state(board, move)
            
        # Process game end
        winner = board.check_winner()
        outcome = 1 if winner == 1 else (-1 if winner == -1 else 0)
        system.end_game(outcome)
        
    # Verify that memory components have learned
    assert len(system.experience.experiences) == 3
    assert len(system.strategic.opening_moves) > 0
    
    # Test that recommendations improve
    board.reset()
    recommendations = system.get_move_recommendation(board)
    assert len(recommendations) > 0
    assert recommendations[0][1] > 0  # Should have positive confidence 