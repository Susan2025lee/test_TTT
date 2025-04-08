import numpy as np
from typing import Tuple, List, Optional

class Board:
    """
    Represents a Tic Tac Toe board.
    Uses numpy array with the following representation:
    0: Empty cell
    1: X (Player 1)
    -1: O (Player 2)
    """
    
    def __init__(self):
        """Initialize an empty board."""
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # X starts
        self.move_history = []
        self.move_count = 0
        
    def make_move(self, position: Tuple[int, int]) -> bool:
        """
        Make a move on the board.
        
        Args:
            position: Tuple of (row, col) coordinates
            
        Returns:
            bool: True if move was valid and made, False otherwise
        """
        row, col = position
        if not self.is_valid_move(position):
            return False
            
        self.board[row, col] = self.current_player
        self.move_history.append((position, self.current_player))
        self.current_player = -self.current_player  # Switch player
        self.move_count += 1
        return True
        
    def is_valid_move(self, position: Tuple[int, int]) -> bool:
        """Check if a move is valid."""
        row, col = position
        if not (0 <= row < 3 and 0 <= col < 3):
            return False
        return self.board[row, col] == 0
        
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves on the current board."""
        return [(i, j) for i in range(3) for j in range(3) 
                if self.board[i, j] == 0]
        
    def check_winner(self) -> Optional[int]:
        """
        Check if there's a winner.
        
        Returns:
            Optional[int]: 1 for X win, -1 for O win, None for no winner
        """
        # Check rows
        for row in range(3):
            if abs(sum(self.board[row])) == 3:
                return self.board[row, 0]
                
        # Check columns
        for col in range(3):
            if abs(sum(self.board[:, col])) == 3:
                return self.board[0, col]
                
        # Check diagonals
        diag_sum = sum(self.board[i, i] for i in range(3))
        if abs(diag_sum) == 3:
            # Return the player ID (1 or -1) based on the sum
            return 1 if diag_sum == 3 else -1
            
        anti_diag_sum = sum(self.board[i, 2-i] for i in range(3))
        if abs(anti_diag_sum) == 3:
            # Return the player ID (1 or -1) based on the sum
            return 1 if anti_diag_sum == 3 else -1
            
        return None
        
    def is_draw(self) -> bool:
        """Check if the game is a draw."""
        return len(self.get_valid_moves()) == 0 and self.check_winner() is None
        
    def is_game_over(self) -> bool:
        """Check if the game is over (win or draw)."""
        return self.check_winner() is not None or self.is_draw()
        
    def get_state(self) -> np.ndarray:
        """Get the current board state."""
        return self.board.copy()
        
    def reset(self) -> None:
        """Reset the board to initial state."""
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        self.move_history = []
        self.move_count = 0
        
    def render(self) -> str:
        """
        Render the board as a string.
        
        Returns:
            str: String representation of the board
        """
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        rows = []
        for i in range(3):
            row = [symbols[self.board[i, j]] for j in range(3)]
            rows.append(f' {row[0]} | {row[1]} | {row[2]} ')
            if i < 2:
                rows.append('-----------')
        return '\n'.join(rows)
        
    def __str__(self) -> str:
        """String representation of the board."""
        return self.render()

    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board()
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        return new_board

    def get_move_count(self) -> int:
        """Get the number of moves made on the board."""
        return self.move_count 