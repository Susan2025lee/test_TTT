"""Core agent implementation for Tic Tac Toe."""

import numpy as np
from typing import Tuple, List, Optional, Dict
from ..environment.board import Board
from ..memory.memory_system import MemorySystem
from ..reward.reward_system import RewardSystem
from ..learning.state_eval import StateEvaluator
from ..learning.action_selection import ActionSelector
from ..learning.experience_processor import ExperienceProcessor

class Agent:
    """Core agent class that integrates all components for decision making."""
    
    def __init__(self,
                 memory_system: MemorySystem,
                 reward_system: RewardSystem,
                 state_evaluator: StateEvaluator,
                 action_selector: ActionSelector,
                 experience_processor: ExperienceProcessor,
                 player_id: int = 1):
        """
        Initialize the agent.
        
        Args:
            memory_system: System for managing different types of memory
            reward_system: System for calculating rewards
            state_evaluator: Component for evaluating board states
            action_selector: Component for selecting moves
            experience_processor: Component for processing and learning from experiences
            player_id: The player's ID (1 for X, -1 for O)
        """
        self.memory_system = memory_system
        self.reward_system = reward_system
        self.state_evaluator = state_evaluator
        self.action_selector = action_selector
        self.experience_processor = experience_processor
        self.player_id = player_id
        
    def select_move(self, board: Board) -> Optional[Tuple[int, int]]:
        """
        Select the next move based on current board state.
        
        Args:
            board: Current game board
            
        Returns:
            Tuple[int, int]: Selected move coordinates
        """
        # --- Remove Explicit Debug Start --- 
        # print(f"DEBUG AGENT: Entry select_move: Player={board.current_player}, State=\\n{board.get_state()}")
        # --- Remove Explicit Debug End --- 

        # Get valid moves
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return None
            
        # Store current state
        current_state = board.get_state()
        self.memory_system.short_term.add_state(board)
        
        # Calculate move values
        move_values = self._evaluate_moves(board, valid_moves)
        
        # Apply strategic considerations
        move_values = self._apply_strategy(board, move_values)
        
        # Select move using action selection policy
        selected_move = self.action_selector.select_move(
            valid_moves=valid_moves,
            move_values=move_values,
            board=board
        )
        
        # Store selected move and update memory
        if selected_move:
            self.memory_system.short_term.add_move(selected_move)
            
            # Update strategic memory with the move
            next_state = current_state.copy()
            row, col = selected_move
            next_state[row, col] = self.player_id
            
            # Extract patterns and update values
            processor = ExperienceProcessor(None, self.memory_system.strategic, None)
            
            # Get patterns before and after move
            current_patterns = processor._extract_patterns(current_state, self.player_id)
            next_patterns = processor._extract_patterns(next_state, self.player_id)
            
            # Update pattern values
            for pattern in next_patterns:
                pattern_str = pattern.split('_')[1]
                if '111' in pattern_str:  # Three in a row
                    self.memory_system.strategic.update_pattern_value(pattern, 0.5)
                elif '110' in pattern_str or '011' in pattern_str or '101' in pattern_str:  # Two in a row
                    self.memory_system.strategic.update_pattern_value(pattern, 0.3)
            
            # Also update patterns that were present before the move
            for pattern in current_patterns:
                pattern_str = pattern.split('_')[1]
                if '11' in pattern_str and '0' in pattern_str:  # Two in a row
                    self.memory_system.strategic.update_pattern_value(pattern, 0.2)
        
        return selected_move
        
    def _evaluate_moves(self, 
                       board: Board,
                       valid_moves: List[Tuple[int, int]]) -> dict:
        """Evaluate moves with exact pattern matching."""
        # --- Remove Debug Print ---
        # print(f"DEBUG _evaluate_moves: Received Player={board.current_player} (type={type(board.current_player)}), State=\\n{board.get_state()}")
        # --- End Remove ---

        # Initialize move values dictionary
        move_values = {move: 0.0 for move in valid_moves}
        state = board.get_state()
        player = board.current_player # Get player from the board passed in
        
        # --- DEBUG PRINT for verification phase state --- 
        # Print the state and player unconditionally
        print(f"\nDEBUG _evaluate_moves: Received Player={player} (type={type(player)}), State=\n{state}")
        
        # Simplified check for the specific state
        verify_state_debug = (state[1,1] == 1 and state[0,0] == -1 and state[2,0] == -1 and np.sum(np.abs(state)) == 3 and player == 1)
        if verify_state_debug:
             print(f"DEBUG: Matched verification state! Evaluating moves:")
        # --- END DEBUG --- 

        for move in valid_moves:
            value = 0.0
            is_win = False # Debug flag
            is_block = False # Debug flag
            
            # Check for immediate win
            is_win = self._is_winning_move(state, move, self.player_id)
            if is_win:
                value = float('inf') # Prioritize winning moves above all else
                move_values[move] = value
                if verify_state_debug:
                     print(f"  DEBUG: Move {move} evaluated as WIN (inf)")
                continue # No need to evaluate further if it's a win

            # Check for immediate block
            is_block = self._is_winning_move(state, move, -self.player_id)
            if is_block:
                 value = 100.0 # High value for blocking opponent's win
                 if verify_state_debug:
                      print(f"  DEBUG: Move {move} evaluated as BLOCK (100.0)")

            # Extract patterns for evaluation (only if not immediate win/loss)
            row_key, col_key, diag_key, anti_diag_key = self._pattern_to_string(state, move)
            row_value = self.memory_system.strategic.get_pattern_value(row_key) if row_key else 0.0
            col_value = self.memory_system.strategic.get_pattern_value(col_key) if col_key else 0.0
            diag_value = 0.0
            if diag_key:
                diag_value = self.memory_system.strategic.get_pattern_value(diag_key)
            anti_diag_value = 0.0
            if anti_diag_key:
                anti_diag_value = self.memory_system.strategic.get_pattern_value(anti_diag_key)
            pattern_based_value = max(abs(row_value), abs(col_value), abs(diag_value), abs(anti_diag_value))
            pos_value = self.memory_system.strategic.get_position_value(move)
            # Combine values (adjust weights as needed)
            calculated_value = pattern_based_value * 2.0 + pos_value * 0.1
            value = max(value, calculated_value) # Max with blocking value if applicable

            move_values[move] = value
            if verify_state_debug:
                 print(f"  DEBUG: Move {move} evaluation details -> Win={is_win}, Block={is_block}, PatternVal={pattern_based_value:.3f}, PosVal={pos_value:.3f}, Calculated={calculated_value:.3f}, Final={value:.3f}")
            
        if verify_state_debug:
             print(f"DEBUG: Final move values: {move_values}")
        return move_values
        
    def _pattern_to_string(self, state: np.ndarray, move: Tuple[int, int]) -> Tuple[str, str, str, str]:
        """Convert pattern to string from player's perspective."""
        row, col = move
        row_pattern = [state[row, i] for i in range(3)]
        row_pattern[col] = self.player_id  # Simulate making the move
        row_str = self._pattern_to_string_helper(row_pattern)
        
        col_pattern = [state[i, col] for i in range(3)]
        col_pattern[row] = self.player_id  # Simulate making the move
        col_str = self._pattern_to_string_helper(col_pattern)
        
        diag_value = 0
        if row == col:
            diag_pattern = [state[i, i] for i in range(3)]
            diag_pattern[row] = self.player_id
            diag_str = self._pattern_to_string_helper(diag_pattern)
            diag_value = self.memory_system.strategic.get_pattern_value(diag_str)
            
        anti_diag_value = 0
        if row + col == 2:
            anti_diag_pattern = [state[i, 2-i] for i in range(3)]
            anti_diag_pattern[2-row] = self.player_id
            anti_diag_str = self._pattern_to_string_helper(anti_diag_pattern)
            anti_diag_value = self.memory_system.strategic.get_pattern_value(anti_diag_str)
        
        return row_str, col_str, diag_str if diag_value > 0 else None, anti_diag_str if anti_diag_value > 0 else None
        
    def _pattern_to_string_helper(self, pattern: List[int]) -> str:
        # Convert to player's perspective (1 for player, 2 for opponent, 0 for empty)
        # This matches ExperienceProcessor._extract_patterns format
        perspective_pattern = []
        for x in pattern:
            if x == self.player_id:
                perspective_pattern.append('1')
            elif x == -self.player_id:
                perspective_pattern.append('2')
            else:
                perspective_pattern.append('0')
        return ''.join(perspective_pattern)
        
    def _is_winning_move(self, state: np.ndarray, move: Tuple[int, int], player: int) -> bool:
        """Check if making the move results in a win for the specified player."""
        row, col = move
        # Check if move is valid (on the board and empty)
        if not (0 <= row < 3 and 0 <= col < 3 and state[row, col] == 0):
            return False # Invalid move cannot be a winning move
        
        # Simulate the move
        next_state = state.copy()
        next_state[row, col] = player
        
        # Check rows
        if np.any(np.sum(next_state, axis=1) == 3 * player):
            return True
        # Check columns
        if np.any(np.sum(next_state, axis=0) == 3 * player):
            return True
        # Check main diagonal
        if np.trace(next_state) == 3 * player:
            return True
        # Check anti-diagonal
        if np.trace(np.fliplr(next_state)) == 3 * player:
            return True
            
        return False
        
    def _creates_fork(self, board: Board, player: int) -> bool:
        """Check if the last move creates a fork (two winning threats)."""
        winning_threats = 0
        for move in board.get_valid_moves():
            test_board = Board()
            test_board.board = board.get_state().copy()
            test_board.current_player = player
            test_board.make_move(move)
            if test_board.check_winner() == player:
                winning_threats += 1
        return winning_threats >= 2
        
    def _apply_strategy(self, board: Board, move_values: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """Apply strategic knowledge to adjust move values."""
        state = board.board
        
        # Get last move from short-term memory
        last_move = None
        if self.memory_system.short_term.get_moves():
            last_move = self.memory_system.short_term.get_moves()[-1]
        
        # If there's a last move, consider counter-moves
        if last_move and last_move in self.memory_system.strategic.counter_moves:
            counters = self.memory_system.strategic.counter_moves[last_move]
            for counter_move, stats in counters.items():
                if counter_move in move_values:
                    # Adjust value based on win rate of counter move
                    win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0.0
                    # Increase value significantly for successful counters
                    move_values[counter_move] += win_rate * 2.0 # Strong boost for good counters
        
        # print("Move values after counter-moves:", move_values)  # Debug
        
        # Apply pattern-based adjustments
        # TODO: Avoid creating processor here if possible, pass needed methods/memory
        
        return move_values
        
    def update_from_game_result(self, outcome: int):
        """
        Update agent's knowledge from game result.
        
        Args:
            outcome: Game outcome (1: win, 0: draw, -1: loss)
        """
        # Get game history
        states = self.memory_system.short_term.get_states()
        moves = self.memory_system.short_term.get_moves()
        
        if not states or not moves:
            return
            
        # Process experience with enhanced pattern learning
        self.experience_processor.process_experience(states, moves, outcome)
        
        # Store in experience memory
        self.memory_system.experience.store_game(states, moves, outcome)
        
        # Clear short-term memory
        self.memory_system.short_term.clear()
        
    def reset(self):
        """Reset the agent's state."""
        self.memory_system.short_term.clear() 