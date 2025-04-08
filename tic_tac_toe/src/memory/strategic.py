import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

class StrategicMemory:
    """
    Strategic memory component that stores and manages winning patterns,
    opening moves, and counter-moves statistics.
    """
    
    def __init__(self):
        """Initialize strategic memory."""
        self.winning_patterns = defaultdict(list)  # Patterns that led to wins
        self.opening_moves = defaultdict(lambda: {'wins': 0, 'total': 0})  # Opening move stats
        self.counter_moves = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'total': 0}))  # Counter-move stats
        self.position_values = self._initialize_position_values()
        self.pattern_values = {}  # Store pattern values
        self.opening_stats = {}   # Store opening move statistics
        
    def _initialize_position_values(self) -> np.ndarray:
        """Initialize position values for the board."""
        return np.array([
            [0.3, 0.2, 0.3],
            [0.2, 0.4, 0.2],
            [0.3, 0.2, 0.3]
        ])
        
    def add_game(self,
                 states: List[np.ndarray],
                 moves: List[Tuple[int, int]],
                 outcome: int):
        """
        Add a complete game to strategic memory.

        Args:
            states: List of board states
            moves: List of moves made
            outcome: Game outcome (1: win, 0: draw, -1: loss)
        """
        if not moves:
            return

        # Update opening move statistics
        self._update_opening_stats(moves[0], outcome)

        # Update counter-move statistics
        self._update_counter_moves(moves, outcome)

        # Store patterns and update pattern values
        from ..learning.experience_processor import ExperienceProcessor
        processor = ExperienceProcessor(None, self, None)  # Temporary processor for pattern extraction
        
        # Process each move in the game
        current_player = 1  # Start with player 1's perspective
        for i, (state, move) in enumerate(zip(states, moves)):
            # Create hypothetical board state after move
            next_state = state.copy()
            row, col = move
            next_state[row, col] = current_player
            
            # Extract patterns from both states for both players
            for player in [1, -1]:
                # Convert outcome to player's perspective
                player_outcome = outcome if player == 1 else -outcome
                
                # Extract patterns from current player's perspective
                current_patterns = processor._extract_patterns(state, player)
                next_patterns = processor._extract_patterns(next_state, player)
                
                # Calculate move value based on outcome and position
                if i == len(states) - 1:  # Last move
                    move_value = player_outcome * 1.0  # Full credit/blame for final move
                elif i == len(states) - 2:  # Second to last move
                    move_value = player_outcome * 0.8  # Strong credit/blame for setup move
                else:
                    # Earlier moves get less credit/blame
                    move_value = player_outcome * (0.5 * (i + 1) / len(states))
                
                # Update pattern values with stronger reinforcement
                for pattern in next_patterns:
                    pattern_str = pattern.split('_')[1]
                    if '111' in pattern_str:  # Three in a row
                        self.update_pattern_value(pattern, move_value * 2.0)  # Double reinforcement for winning patterns
                    elif '110' in pattern_str or '011' in pattern_str or '101' in pattern_str:  # Two in a row
                        self.update_pattern_value(pattern, move_value * 1.5)  # Strong reinforcement for near-wins
                    else:
                        self.update_pattern_value(pattern, move_value * 0.5)  # Base reinforcement for other patterns
                
                # Also update patterns that were present before the move
                for pattern in current_patterns:
                    pattern_str = pattern.split('_')[1]
                    if '11' in pattern_str and '0' in pattern_str:  # Two in a row
                        self.update_pattern_value(pattern, move_value * 1.0)  # Normal reinforcement for existing patterns
            
            # Switch players for next move
            current_player = -current_player
        
    def _update_opening_stats(self, move: Tuple[int, int], outcome: int):
        """Update statistics for opening moves."""
        self.opening_moves[move]['total'] += 1
        if outcome == 1:
            self.opening_moves[move]['wins'] += 1
            
    def _update_counter_moves(self, moves: List[Tuple[int, int]], outcome: int):
        """Update statistics for counter-moves."""
        for i in range(len(moves) - 1):
            move = moves[i]
            counter = moves[i + 1]
            self.counter_moves[move][counter]['total'] += 1
            if outcome == 1:
                self.counter_moves[move][counter]['wins'] += 1
                
    def _store_winning_pattern(self, 
                             states: List[np.ndarray],
                             moves: List[Tuple[int, int]]):
        """Store a winning pattern."""
        for state, move in zip(states, moves):
            pattern_key = self._create_pattern_key(state)
            self.winning_patterns[pattern_key].append(move)
            
    def _create_pattern_key(self, state: np.ndarray) -> str:
        """Create a string key for a board pattern."""
        return ','.join(map(str, state.flatten()))
        
    def get_best_opening(self) -> Optional[Tuple[int, int]]:
        """Get the best opening move based on statistics."""
        if not self.opening_moves:
            return None
            
        best_move = max(
            self.opening_moves.items(),
            key=lambda x: (x[1]['wins'] / x[1]['total'] if x[1]['total'] > 0 else 0)
        )
        return best_move[0]
        
    def get_best_counter(self, 
                        opponent_move: Tuple[int, int],
                        n: int = 3) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get the best counter-moves for an opponent's move.
        
        Args:
            opponent_move: The opponent's move to counter
            n: Number of counter-moves to return
            
        Returns:
            List of (move, win_rate) tuples
        """
        if opponent_move not in self.counter_moves:
            return []
            
        counters = self.counter_moves[opponent_move]
        if not counters:
            return []
            
        counter_rates = [
            (move, stats['wins'] / stats['total'] if stats['total'] > 0 else 0)
            for move, stats in counters.items()
        ]
        
        return sorted(counter_rates, key=lambda x: x[1], reverse=True)[:n]
        
    def get_winning_moves(self, state: np.ndarray) -> List[Tuple[int, int]]:
        """Get potential winning moves from current state."""
        winning_moves = []
        for row in range(3):
            for col in range(3):
                if state[row, col] == 0:  # Empty position
                    # Try move
                    next_state = state.copy()
                    next_state[row, col] = 1
                    
                    # Extract patterns
                    from ..learning.experience_processor import ExperienceProcessor
                    processor = ExperienceProcessor(None, self, None)
                    patterns = processor._extract_patterns(next_state, 1)
                    
                    # Check for winning or high-value patterns
                    for pattern in patterns:
                        pattern_str = pattern.split('_')[1]
                        if '111' in pattern_str or self.get_pattern_value(pattern) > 0.8:
                            winning_moves.append((row, col))
                            break
        
        return winning_moves
        
    def get_position_value(self, position: Tuple[int, int]) -> float:
        """Get the strategic value of a board position."""
        row, col = position
        value = self.position_values[row, col]
        # Remove debug print
        # print(f"DEBUG get_position_value: Pos={position}, Value={value:.3f}")
        return value
        
    def update_position_value(self, position: Tuple[int, int], outcome: int):
        """Update the strategic value of a position based on game outcome."""
        row, col = position
        if outcome == 1:  # Win
            self.position_values[row, col] *= 1.1  # Increase value by 10%
        elif outcome == -1:  # Loss
            self.position_values[row, col] *= 0.9  # Decrease value by 10%
            
    def get_strategic_value(self, 
                          move: Tuple[int, int],
                          state: np.ndarray) -> float:
        """
        Calculate the strategic value of a move in a given state.
        
        Args:
            move: The move to evaluate
            state: Current board state
            
        Returns:
            float: Strategic value of the move
        """
        # Base position value
        value = self.get_position_value(move)
        
        # Add opening move bonus
        if np.sum(np.abs(state)) == 0:  # First move
            opening_stats = self.opening_moves[move]
            if opening_stats['total'] > 0:
                value += 0.2 * (opening_stats['wins'] / opening_stats['total'])
                
        # Add winning pattern bonus
        if move in self.get_winning_moves(state):
            value += 0.3
            
        return value
        
    def update_pattern_value(self, pattern: str, value_change: float):
        """Update the value of a pattern and ALL its equivalents."""
        # Split pattern into type and sequence, handling anti_diag case
        parts = pattern.split('_')
        if len(parts) == 3 and parts[0] == 'anti':
            pattern_type = 'anti_diag'
            pattern_seq = parts[2]
        else:
            pattern_type = parts[0]
            pattern_seq = parts[1]

        # Get all equivalent patterns once
        equivalent_patterns = self._get_equivalent_patterns_non_recursive(pattern_type, pattern_seq)

        # The value_change coming in is already scaled by ExperienceProcessor
        base_update = value_change
        calculated_new_value = None # Store the calculated value to apply to all equivalents

        # Calculate the new value based on the first equivalent (or the pattern itself)
        # Assumption: All equivalents should converge to the same value
        pattern_to_calculate_from = next(iter(equivalent_patterns)) 

        if pattern_to_calculate_from not in self.pattern_values:
            # Initialize with a small value in the direction of base_update
            initial_value = 0.1 if base_update > 0 else -0.1 
            current_value = initial_value
        else:
            current_value = self.pattern_values[pattern_to_calculate_from]
        
        # Apply adaptive momentum: larger changes when value is small or update is significant
        momentum = min(1.0, (1.0 - abs(current_value)) + abs(base_update) * 0.5)
        calculated_new_value = current_value + base_update * momentum

        # Normalize to [-1, 1] with soft bounds
        if abs(calculated_new_value) > 1.0:
            calculated_new_value = np.sign(calculated_new_value) * (1.0 - 0.1 * np.exp(-5 * (abs(calculated_new_value) - 1.0)))

        # Update ALL equivalent patterns in the dictionary with the SAME calculated value
        for equiv_pattern in equivalent_patterns:
            self.pattern_values[equiv_pattern] = calculated_new_value

        # Update related patterns without recursion using the scaled value_change
        # Note: This propagation might now update patterns already set above,
        # but _update_related_patterns should handle this gracefully.
        self._update_related_patterns_non_recursive(pattern_type, pattern_seq, value_change)

    def _update_related_patterns_non_recursive(self, pattern_type: str, pattern_seq: str, value_change: float):
        """Update patterns that are related to the current pattern without recursion, applying updates directly."""
        # The value_change passed here is the original, scaled value from ExperienceProcessor

        # Function to apply update directly to a pattern and its equivalents
        def apply_direct_update(target_pattern_type, target_pattern_seq, update_value):
            target_equivalents = self._get_equivalent_patterns_non_recursive(target_pattern_type, target_pattern_seq)
            for equiv in target_equivalents:
                if equiv not in self.pattern_values:
                    current_value = 0.1 if update_value > 0 else -0.1
                else:
                    current_value = self.pattern_values[equiv]
                
                momentum = min(1.0, (1.0 - abs(current_value)) + abs(update_value) * 0.5)
                new_value = current_value + update_value * momentum
                
                if abs(new_value) > 1.0:
                    new_value = np.sign(new_value) * (1.0 - 0.1 * np.exp(-5 * (abs(new_value) - 1.0)))
                
                # Directly update the dictionary value for this equivalent
                self.pattern_values[equiv] = new_value

        # For winning patterns (111 or 222), update setup patterns
        if pattern_seq == '111' or pattern_seq == '222':
            propagation_update = value_change * 0.5 # Scaled update for setups
            for i in range(3):
                setup_seq = list(pattern_seq)
                setup_seq[i] = '0'
                setup_pattern_seq = ''.join(setup_seq)
                # Apply update directly to setup pattern and its equivalents
                apply_direct_update(pattern_type, setup_pattern_seq, propagation_update)

        # For two-in-a-row patterns, update patterns that could complete the win
        elif '110' in pattern_seq or '011' in pattern_seq or '101' in pattern_seq:
             empty_pos = pattern_seq.find('0') 
             if empty_pos != -1:
                 player_piece = '1' if '1' in pattern_seq else '2'
                 complete_seq = list(pattern_seq)
                 complete_seq[empty_pos] = player_piece
                 complete_pattern_seq = ''.join(complete_seq)
                 propagation_update = value_change * 0.3 # Smaller scaled update for completions
                 # Apply update directly to completion pattern and its equivalents
                 apply_direct_update(pattern_type, complete_pattern_seq, propagation_update)

    def _get_equivalent_patterns_non_recursive(self, pattern_type: str, pattern_seq: str) -> set[str]:
        """Get all equivalent patterns through rotations and reflections without recursion."""
        patterns = set()
        
        # Add the original pattern
        if pattern_type == 'anti_diag':
            patterns.add(f'anti_diag_{pattern_seq}')
        else:
            patterns.add(f'{pattern_type}_{pattern_seq}')

        # Add rotations for rows/columns/diagonals
        if pattern_type in ['row', 'col']:
            # Row becomes column and vice versa
            patterns.add(f'{"col" if pattern_type == "row" else "row"}_{pattern_seq}')
        elif pattern_type == 'diag':
            # Main diagonal can become anti-diagonal
            patterns.add(f'anti_diag_{pattern_seq}')
        elif pattern_type == 'anti_diag':
            # Anti-diagonal can become main diagonal
            patterns.add(f'diag_{pattern_seq}')

        # Add pattern with reversed sequence (reflection)
        rev_seq = pattern_seq[::-1]
        if pattern_type == 'anti_diag':
            patterns.add(f'anti_diag_{rev_seq}')
        else:
            patterns.add(f'{pattern_type}_{rev_seq}')

        return patterns
        
    def get_pattern_value(self, pattern: str) -> float:
        """Get the value of a pattern. Assumes equivalents are already stored."""
        value = self.pattern_values.get(pattern, 0.0) # Default to 0.0 if pattern not found
        # Remove debug print
        # print(f"DEBUG get_pattern_value: Pattern='{pattern}', Value={value:.3f}")
        return value
        
    def clear(self):
        """Clear all strategic memory."""
        self.winning_patterns.clear()
        self.opening_moves.clear()
        self.counter_moves.clear()
        self.position_values = self._initialize_position_values()
        self.pattern_values.clear() 