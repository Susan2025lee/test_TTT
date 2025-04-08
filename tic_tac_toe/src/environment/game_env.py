from typing import Tuple, Dict, Optional, List
import numpy as np
from .board import Board

class GameEnvironment:
    """
    Tic Tac Toe game environment that manages game state and rules.
    Provides an interface for agents to interact with the game.
    """
    
    def __init__(self):
        """Initialize the game environment."""
        self.board = Board()
        self.reset()
        
    def reset(self) -> np.ndarray:
        """
        Reset the game environment to initial state.
        
        Returns:
            np.ndarray: Initial board state
        """
        self.board.reset()
        return self.get_state()
        
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment by making a move.
        
        Args:
            action: Tuple of (row, col) coordinates for the move
            
        Returns:
            Tuple containing:
            - np.ndarray: New board state
            - float: Reward for the action
            - bool: Whether the game is done
            - Dict: Additional information
        """
        # Make the move
        valid_move = self.board.make_move(action)
        if not valid_move:
            return self.get_state(), -10.0, True, {'error': 'Invalid move'}
            
        # Check game status
        winner = self.board.check_winner()
        done = self.board.is_game_over()
        
        # Calculate reward
        reward = self._calculate_reward(winner, done)
        
        # Get additional info
        info = {
            'winner': winner,
            'is_draw': self.board.is_draw(),
            'valid_moves': self.get_valid_moves()
        }
        
        return self.get_state(), reward, done, info
        
    def _calculate_reward(self, winner: Optional[int], done: bool) -> float:
        """
        Calculate the reward for the current game state.
        
        Args:
            winner: The winner of the game (1, -1, or None)
            done: Whether the game is finished
            
        Returns:
            float: Calculated reward
        """
        if not done:
            return 0.0
            
        if winner is None:  # Draw
            return 0.5
            
        return 1.0 if winner == self.board.current_player else -1.0
        
    def get_state(self) -> np.ndarray:
        """Get the current game state."""
        return self.board.get_state()
        
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get list of valid moves."""
        return self.board.get_valid_moves()
        
    def is_valid_move(self, action: Tuple[int, int]) -> bool:
        """Check if a move is valid."""
        return self.board.is_valid_move(action)
        
    def get_current_player(self) -> int:
        """Get the current player (1 for X, -1 for O)."""
        return self.board.current_player
        
    def render(self) -> str:
        """Render the current game state as a string."""
        return str(self.board)
        
    def get_move_history(self) -> List[Tuple[Tuple[int, int], int]]:
        """Get the history of moves made in the game."""
        return self.board.move_history.copy()
        
    def clone(self) -> 'GameEnvironment':
        """Create a deep copy of the current game environment."""
        new_env = GameEnvironment()
        new_env.board.board = self.board.board.copy()
        new_env.board.current_player = self.board.current_player
        new_env.board.move_history = self.board.move_history.copy()
        return new_env
        
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.is_game_over()
        
    def get_winner(self) -> Optional[int]:
        """Get the winner of the game."""
        return self.board.check_winner()
        
    def get_board(self) -> Board:
        """Get the current game board."""
        return self.board
        
    def make_move(self, move: Tuple[int, int]) -> bool:
        """Make a move on the board."""
        return self.board.make_move(move) 