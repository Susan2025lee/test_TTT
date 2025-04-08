"""Action selection module for the Tic Tac Toe agent."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import math
import random

from ..environment import Board
from .state_eval import StateEvaluator

class ActionSelector:
    """Selects actions based on various exploration strategies."""
    
    def __init__(self, state_evaluator: StateEvaluator, epsilon: float = 0.1, 
                 temperature: float = 1.0, ucb_constant: float = 1.0):
        """
        Initialize the action selector.
        
        Args:
            state_evaluator: Evaluator for board states
            epsilon: Exploration rate for epsilon-greedy strategy (0 to 1)
            temperature: Temperature for softmax sampling (> 0)
            ucb_constant: Exploration constant for UCB (> 0)
        """
        self.state_evaluator = state_evaluator
        self.epsilon = epsilon
        self.temperature = temperature
        self.ucb_constant = ucb_constant
        
        # Visit counts for UCB
        self.visit_counts: Dict[str, Dict[Tuple[int, int], int]] = {}
        self.total_visits: Dict[str, int] = {}
        
    def select_move(self,
                   valid_moves: List[Tuple[int, int]],
                   move_values: Dict[Tuple[int, int], float],
                   board: Board) -> Optional[Tuple[int, int]]:
        """
        Select a move using pattern-based evaluation.
        
        Args:
            valid_moves: List of valid moves
            move_values: Dictionary of move values
            board: Current board state
            
        Returns:
            Selected move coordinates
        """
        if not valid_moves:
            return None
            
        # Priority 1: Take any move evaluated as an immediate win (value=inf)
        winning_moves = [move for move, value in move_values.items() if value == float('inf')]
        if winning_moves:
            # If multiple winning moves (shouldn't happen in TicTacToe?), pick the first
            return winning_moves[0]
            
        # Priority 2: Block opponent's immediate win (value=100.0)
        # Use a high finite value check for blocks
        blocking_moves = [move for move, value in move_values.items() if value == 100.0]
        if blocking_moves:
            # If multiple blocking moves, pick the first
            return blocking_moves[0]
            
        # Priority 3: Early game strategy (center -> corner)
        if board.get_move_count() < 2:
            center = (1, 1)
            if center in valid_moves:
                return center
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            available_corners = [c for c in corners if c in valid_moves]
            if available_corners:
                corner = random.choice(available_corners)
                return corner
                
        # Priority 4: Take moves based on highest evaluated value (excluding inf/100)
        # Filter out win/block moves already handled
        evaluable_moves = {move: value for move, value in move_values.items() if value < 100.0}
        if not evaluable_moves:
             # Should not happen if valid_moves is not empty and no win/block
             # If it does, fall back to random choice among remaining valid moves
             remaining_valid = [m for m in valid_moves if m not in winning_moves and m not in blocking_moves]
             return random.choice(remaining_valid) if remaining_valid else None

        best_move = max(evaluable_moves.items(), key=lambda item: item[1])[0]
        return best_move
        
    def select_move_ucb(self, board: Board) -> Tuple[int, int]:
        """
        Select a move using UCB strategy.
        
        Args:
            board: Current game board
            
        Returns:
            Selected move as (row, col)
        """
        valid_moves = board.get_valid_moves()
        board_state = self._get_board_state_key(board)
        
        # Initialize visit counts if needed
        if board_state not in self.visit_counts:
            self.visit_counts[board_state] = {move: 0 for move in valid_moves}
            self.total_visits[board_state] = 0
            
            # For initial state, evaluate all moves once
            move_values = {move: self._evaluate_move(board, move) for move in valid_moves}
            best_move = max(move_values.items(), key=lambda x: x[1])[0]
            self.visit_counts[board_state][best_move] = 1
            self.total_visits[board_state] = 1
            return best_move
            
        # Calculate UCB values for each move
        ucb_values = {}
        for move in valid_moves:
            value = self._evaluate_move(board, move)
            visit_count = self.visit_counts[board_state][move]
            total_visits = self.total_visits[board_state]
            
            if visit_count == 0:
                # For unvisited moves, use value plus exploration bonus
                ucb_values[move] = value + self.ucb_constant
            else:
                # UCB1 formula: value + c * sqrt(ln(N) / n)
                exploration_term = self.ucb_constant * np.sqrt(
                    np.log(total_visits) / visit_count
                )
                ucb_values[move] = value + exploration_term
                
        # Select move with highest UCB value
        best_move = max(ucb_values.items(), key=lambda x: x[1])[0]
        
        # Update visit counts
        self.visit_counts[board_state][best_move] += 1
        self.total_visits[board_state] += 1
        
        return best_move
        
    def select_move_temperature(self, board: Board) -> Tuple[int, int]:
        """Select a move using temperature-based sampling."""
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves")

        # Get values for all moves
        move_values = {move: self._evaluate_move(board, move) for move in valid_moves}
        
        # Scale values to be non-negative
        min_value = min(move_values.values())
        scaled_values = {move: value - min_value + 1e-6 for move, value in move_values.items()}
        
        # Convert to probabilities using softmax
        total = sum(math.exp(value / self.temperature) for value in scaled_values.values())
        probabilities = {
            move: math.exp(value / self.temperature) / total 
            for move, value in scaled_values.items()
        }
        
        # Sample move based on probabilities
        moves = list(probabilities.keys())
        probs = list(probabilities.values())
        return random.choices(moves, weights=probs)[0]
        
    def get_move_probabilities(self, board: Board) -> Dict[Tuple[int, int], float]:
        """
        Calculate probability distribution over moves.
        
        Args:
            board: Current game board
            
        Returns:
            Dictionary mapping moves to their probabilities
        """
        valid_moves = board.get_valid_moves()
        move_values = {move: self._evaluate_move(board, move) for move in valid_moves}
        
        # Apply temperature scaling
        values = np.array(list(move_values.values()))
        moves = list(move_values.keys())
        
        # Softmax with temperature
        scaled_values = values / self.temperature
        exp_values = np.exp(scaled_values - np.max(scaled_values))
        probabilities = exp_values / np.sum(exp_values)
        
        return dict(zip(moves, probabilities))
        
    def _evaluate_move(self, board: Board, move: Tuple[int, int]) -> float:
        """Evaluate the potential value of a single move.
        
        Combines line potential, pattern contribution, and basic position value.
        Does NOT check for immediate win/block, assuming that's handled elsewhere.
        """
        # Use evaluator's helper methods to assess move potential
        line_potential = self.state_evaluator._assess_line_potential(board, move)
        pattern_value = self.state_evaluator._assess_pattern_value(board, move)

        # Basic positional value (e.g., center > corner > edge)
        # Give center significantly higher base value for deterministic selection
        if move == self.state_evaluator.center:
            base_pos_value = 0.6 # Significantly higher than others
        elif move in self.state_evaluator.corners:
            base_pos_value = 0.2
        else: # Edges
            base_pos_value = 0.1
            
        # Combine the scores (weights can be tuned)
        # Example weighting: 50% line, 30% pattern, 20% base position
        total_value = 0.5 * line_potential + 0.3 * pattern_value + 0.2 * base_pos_value
        
        # Ensure value is within a reasonable range if necessary (e.g., 0 to 1)
        return max(0.0, min(1.0, total_value))
        
    def _get_best_move(self, board: Board) -> Tuple[int, int]:
        """Get the move with the highest evaluation."""
        valid_moves = board.get_valid_moves()
        move_values = {move: self._evaluate_move(board, move) for move in valid_moves}
        return max(move_values.items(), key=lambda x: x[1])[0]
        
    def _get_board_state_key(self, board: Board) -> str:
        """Convert board state to a string key for visit count tracking."""
        return str(board.get_state().tolist()) 