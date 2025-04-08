import numpy as np
from typing import List, Tuple, Dict, Optional
from ..environment.board import Board

class ShortTermMemory:
    """
    Short-term memory component that stores and tracks the current game state
    and recent move history.
    """
    
    def __init__(self, max_history_length: int = 50):
        """
        Initialize short-term memory.
        
        Args:
            max_history_length: Maximum number of board states to keep in history
        """
        self.max_history_length = max_history_length
        self.reset()
        
    def reset(self):
        """Reset the memory to initial state."""
        self.current_sequence = []  # List of moves in current game
        self.board_states = []      # History of board states
        self.player_states = []     # History of player turns
        self.move_scores = {}       # Scores for moves in current game
        self.moves = []             # List of moves made
        
    def add_state(self, board: Board, move: Optional[Tuple[int, int]] = None, score: float = 0.0):
        """
        Add a new state to memory.
        
        Args:
            board: Current game board
            move: The move that led to this state (if any)
            score: Score/value associated with this state
        """
        # Store board state
        self.board_states.append(board.get_state().copy())
        self.player_states.append(board.current_player)
        
        # Store move if provided
        if move is not None:
            self.current_sequence.append(move)
            self.move_scores[move] = score
            self.moves.append(move)
            
        # Maintain history length
        if len(self.board_states) > self.max_history_length:
            self.board_states.pop(0)
            self.player_states.pop(0)
        
    def add_move(self, move: Tuple[int, int]):
        """Add a move to the move history."""
        self.moves.append(move)
        
    def get_last_move(self) -> Optional[Tuple[int, int]]:
        """Get the last move made."""
        if not self.moves:
            return None
        return self.moves[-1]
        
    def get_moves(self) -> List[Tuple[int, int]]:
        """Get all moves made."""
        return self.moves.copy()
        
    def get_states(self) -> List[np.ndarray]:
        """Get all board states."""
        return self.board_states.copy()
        
    def get_current_sequence(self) -> List[Tuple[int, int]]:
        """Get the sequence of moves in the current game."""
        return self.current_sequence.copy()
        
    def get_last_n_states(self, n: int) -> List[np.ndarray]:
        """Get the last n board states."""
        return self.board_states[-n:].copy()
        
    def get_move_score(self, move: Tuple[int, int]) -> float:
        """Get the score associated with a move."""
        return self.move_scores.get(move, 0.0)
        
    def get_state_history(self) -> List[np.ndarray]:
        """Get the complete state history."""
        return self.board_states.copy()
        
    def get_context(self, n_recent: int = 3) -> dict:
        """
        Get the current game context.
        
        Args:
            n_recent: Number of recent states to include
            
        Returns:
            dict: Context information including recent states and moves
        """
        recent_states = self.board_states[-n_recent:]
        recent_players = self.player_states[-n_recent:]
        
        return {
            'recent_states': recent_states,
            'recent_players': recent_players,
            'current_sequence': self.current_sequence,
            'move_scores': self.move_scores.copy()
        }
        
    def update_move_score(self, move: Tuple[int, int], score: float):
        """Update the score for a specific move."""
        self.move_scores[move] = score
        
    def get_best_move(self) -> Optional[Tuple[int, int]]:
        """Get the move with highest score."""
        if not self.move_scores:
            return None
        return max(self.move_scores.items(), key=lambda x: x[1])[0]

    def clear(self):
        """Clear all stored states and moves."""
        self.reset() 