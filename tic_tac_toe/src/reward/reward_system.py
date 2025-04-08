import numpy as np
from typing import Tuple, Dict, Optional
from ..environment.board import Board

class RewardSystem:
    """
    Reward system that calculates immediate, terminal, and shaped rewards
    for the Tic Tac Toe agent.
    """
    
    def __init__(self):
        """Initialize the reward system with default reward values."""
        # Terminal rewards
        self.win_reward = 1.0
        self.loss_reward = -1.0
        self.draw_reward = 0.0
        self.invalid_move_reward = -10.0
        
        # Position-based rewards
        self.position_values = np.array([
            [0.3, 0.2, 0.3],  # Corners and edges values
            [0.2, 0.4, 0.2],  # Center has highest value
            [0.3, 0.2, 0.3]
        ])
        
        # Tactical rewards
        self.blocking_reward = 0.2
        self.winning_opportunity_reward = 0.3
        self.fork_creation_reward = 0.4
        
    def get_immediate_reward(self,
                           board: Board,
                           move: Tuple[int, int],
                           player: int) -> float:
        """
        Get the immediate reward for a move.
        
        Args:
            board: Current game board
            move: The move to evaluate
            player: The player making the move
            
        Returns:
            float: Immediate reward value
        """
        # Store current player and temporarily set to the input player
        current_player = board.current_player
        board.current_player = player
        
        # Calculate reward
        total_reward, _ = self.calculate_reward(board, move)
        
        # Restore original player
        board.current_player = current_player
        
        return total_reward
        
    def calculate_reward(self,
                        board: Board,
                        move: Tuple[int, int],
                        is_terminal: bool = False) -> Tuple[float, Dict]:
        """
        Calculate the total reward for a move.
        
        Args:
            board: Current game board
            move: The move made (row, col)
            is_terminal: Whether this is a terminal state
            
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        rewards = {}
        
        # Calculate terminal rewards first
        if is_terminal:
            terminal_reward = self._calculate_terminal_reward(board)
            rewards['terminal'] = terminal_reward
            return terminal_reward, rewards
            
        # Check if move is valid
        if not board.is_valid_move(move):
            rewards['invalid_move'] = self.invalid_move_reward
            return self.invalid_move_reward, rewards
            
        # Calculate immediate position-based reward
        position_reward = self._calculate_position_reward(move)
        rewards['position'] = position_reward
        
        # Calculate tactical rewards
        tactical_reward = self._calculate_tactical_reward(board, move)
        rewards['tactical'] = tactical_reward
        
        # Calculate progress reward (shaped reward)
        progress_reward = self._calculate_progress_reward(board, move)
        rewards['progress'] = progress_reward
        
        # Sum up all rewards
        total_reward = sum(rewards.values())
        
        return total_reward, rewards
        
    def _calculate_terminal_reward(self, board: Board) -> float:
        """Calculate reward for terminal states."""
        winner = board.check_winner()
        if winner is None:
            if board.is_draw():
                return self.draw_reward
            return 0.0
            
        return self.win_reward if winner == board.current_player else self.loss_reward
        
    def _calculate_position_reward(self, move: Tuple[int, int]) -> float:
        """Calculate reward based on move position."""
        row, col = move
        return self.position_values[row, col]
        
    def _calculate_tactical_reward(self, board: Board, move: Tuple[int, int]) -> float:
        """Calculate reward for tactical moves."""
        reward = 0.0
        
        # Make a copy of the board to simulate the move
        board_copy = board.get_state()
        row, col = move
        board_copy[row, col] = board.current_player
        
        # Check if move blocks opponent's win
        if self._blocks_opponent_win(board, move):
            reward += self.blocking_reward
            
        # Check if move creates a winning opportunity
        if self._creates_winning_opportunity(board_copy, board.current_player):
            reward += self.winning_opportunity_reward
            
        # Check if move creates a fork (two winning paths)
        if self._creates_fork(board_copy, board.current_player):
            reward += self.fork_creation_reward
            
        return reward
        
    def _calculate_progress_reward(self, board: Board, move: Tuple[int, int]) -> float:
        """Calculate reward based on game progress."""
        # Count number of pieces on board
        num_pieces = np.count_nonzero(board.get_state())
        
        # Early game: reward center and corner control
        if num_pieces < 3:
            row, col = move
            if (row, col) == (1, 1):  # Center
                return 0.2
            if row in [0, 2] and col in [0, 2]:  # Corners
                return 0.15
                
        # Mid game: reward creating opportunities
        elif num_pieces < 6:
            if self._creates_winning_opportunity(board.get_state(), board.current_player):
                return 0.1
                
        # Late game: reward blocking opponent
        else:
            if self._blocks_opponent_win(board, move):
                return 0.25
                
        return 0.0
        
    def _blocks_opponent_win(self, board: Board, move: Tuple[int, int]) -> bool:
        """Check if move blocks opponent's winning move."""
        # Make a copy of the board
        board_copy = board.get_state()
        row, col = move
        
        # Try opponent's move in this position
        board_copy[row, col] = -board.current_player
        
        # Check if this would have been a win for opponent
        lines = [
            board_copy[row],  # Row
            board_copy[:, col],  # Column
            np.diagonal(board_copy),  # Main diagonal
            np.diagonal(np.fliplr(board_copy))  # Anti-diagonal
        ]
        
        return any(abs(sum(line)) == 3 for line in lines)
        
    def _creates_winning_opportunity(self, board_state: np.ndarray, player: int) -> bool:
        """Check if move creates a winning opportunity."""
        # Check each line (rows, columns, diagonals)
        for i in range(3):
            # Check rows and columns
            if sum(board_state[i]) == player * 2:
                return True
            if sum(board_state[:, i]) == player * 2:
                return True
                
        # Check diagonals
        if sum(np.diagonal(board_state)) == player * 2:
            return True
        if sum(np.diagonal(np.fliplr(board_state))) == player * 2:
            return True
            
        return False
        
    def _creates_fork(self, board_state: np.ndarray, player: int) -> bool:
        """Check if move creates a fork (two winning paths)."""
        winning_paths = 0
        
        # Check rows
        for row in board_state:
            if sum(row == player) == 2 and sum(row == 0) == 1:
                winning_paths += 1
                
        # Check columns
        for col in board_state.T:
            if sum(col == player) == 2 and sum(col == 0) == 1:
                winning_paths += 1
                
        # Check diagonals
        diag = np.diagonal(board_state)
        if sum(diag == player) == 2 and sum(diag == 0) == 1:
            winning_paths += 1
            
        anti_diag = np.diagonal(np.fliplr(board_state))
        if sum(anti_diag == player) == 2 and sum(anti_diag == 0) == 1:
            winning_paths += 1
            
        return winning_paths >= 2
        
    def update_rewards(self, game_outcome: int):
        """
        Update reward values based on game outcome.
        This can be used to dynamically adjust rewards based on agent performance.
        
        Args:
            game_outcome: The outcome of the game (1: win, 0: draw, -1: loss)
        """
        adjustment = 0.05  # Base adjustment value
        
        # Adjust tactical rewards based on outcome
        if game_outcome == 1:  # Win
            self.blocking_reward -= adjustment  # Decrease defensive play
            self.winning_opportunity_reward += adjustment  # Increase aggressive play
        elif game_outcome == -1:  # Loss
            self.blocking_reward += adjustment  # Increase defensive play
            self.winning_opportunity_reward -= adjustment  # Decrease aggressive play
            
        # Ensure rewards stay within reasonable bounds
        self.blocking_reward = np.clip(self.blocking_reward, 0.1, 0.5)
        self.winning_opportunity_reward = np.clip(self.winning_opportunity_reward, 0.1, 0.5)
        self.fork_creation_reward = np.clip(self.fork_creation_reward, 0.2, 0.6) 