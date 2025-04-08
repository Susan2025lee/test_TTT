import numpy as np
from typing import List, Tuple, Dict, Optional
from ..environment.board import Board
from ..memory.experience import ExperienceMemory
from ..memory.strategic import StrategicMemory
from .state_eval import StateEvaluator

class ExperienceProcessor:
    """Processes and learns from game experiences to improve agent performance."""
    
    def __init__(self, 
                 experience_memory: ExperienceMemory,
                 strategic_memory: StrategicMemory,
                 state_evaluator: StateEvaluator,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9):
        """Initialize the experience processor."""
        self.experience_memory = experience_memory
        self.strategic_memory = strategic_memory
        self.state_evaluator = state_evaluator
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.value_estimates = {}
        self._pattern_cache: Dict[Tuple[int, bytes], List[str]] = {} # Cache for _extract_patterns
        self._cache_hits = 0
        self._cache_misses = 0
        
    def __del__(self):
        # Print cache stats on deletion (useful for seeing effectiveness)
        # Commented out for release
        # total_calls = self._cache_hits + self._cache_misses
        # if total_calls > 0:
        #     hit_rate = (self._cache_hits / total_calls) * 100
        #     print(f"ExperienceProcessor _pattern_cache stats: Hits={self._cache_hits}, Misses={self._cache_misses}, Hit Rate={hit_rate:.2f}%")
        # else:
        #     print("ExperienceProcessor _pattern_cache stats: No calls recorded.")
        pass # Keep __del__ empty or remove if nothing else needed
        
    def process_experience(self, states: List[np.ndarray], moves: List[Tuple[int, int]], outcome: int):
        """Process experience from a game."""
        # Calculate returns for value learning
        returns = self._calculate_returns(len(states), outcome)

        # Process each state-move pair
        for i, (state, move) in enumerate(zip(states, moves)):
            # Get patterns for the current state
            current_player = 1 if i % 2 == 0 else -1
            patterns = self._extract_patterns(state, current_player)
            opp_patterns = self._extract_patterns(state, -current_player)

            # Make the move to get next state
            next_state = state.copy()
            next_state[move[0], move[1]] = current_player
            
            # Get patterns after the move
            next_patterns = self._extract_patterns(next_state, current_player)
            next_opp_patterns = self._extract_patterns(next_state, -current_player)

            # Calculate move value based on position in sequence and outcome
            move_value = returns[i] * (1.0 if current_player == 1 else -1.0)
            if i == len(states) - 1:  # Last move
                move_value *= 3.0  # Triple impact for winning move
            elif i == len(states) - 2:  # Second to last move
                move_value *= 2.0  # Double impact for setup to win
            elif i == len(states) - 3:  # Third to last move
                move_value *= 1.5  # Stronger impact for early setup

            # Update pattern values with stronger reinforcement for winning patterns
            for pattern in next_patterns:
                pattern_str = pattern.split('_')[1] if '_' in pattern else ''
                if '111' in pattern_str or '222' in pattern_str:  # Three in a row
                    self.strategic_memory.update_pattern_value(pattern, move_value * 2.0)
                elif '110' in pattern_str or '011' in pattern_str or '101' in pattern_str:  # Two in a row
                    self.strategic_memory.update_pattern_value(pattern, move_value * 1.5)
                else:
                    self.strategic_memory.update_pattern_value(pattern, move_value * 0.5)
            
            # Update opponent's pattern values with inverse reward
            for pattern in next_opp_patterns:
                pattern_str = pattern.split('_')[1] if '_' in pattern else ''
                if '111' in pattern_str or '222' in pattern_str:  # Three in a row
                    self.strategic_memory.update_pattern_value(pattern, -move_value * 2.0)
                elif '110' in pattern_str or '011' in pattern_str or '101' in pattern_str:  # Two in a row
                    self.strategic_memory.update_pattern_value(pattern, -move_value * 1.5)
                else:
                    self.strategic_memory.update_pattern_value(pattern, -move_value * 0.5)

            # Update patterns that were present before the move
            for pattern in patterns:
                pattern_str = pattern.split('_')[1] if '_' in pattern else ''
                if '11' in pattern_str or '22' in pattern_str:  # Two in a row
                    self.strategic_memory.update_pattern_value(pattern, move_value * 0.8)
            
            for pattern in opp_patterns:
                pattern_str = pattern.split('_')[1] if '_' in pattern else ''
                if '11' in pattern_str or '22' in pattern_str:  # Two in a row
                    self.strategic_memory.update_pattern_value(pattern, -move_value * 0.8)

    def _calculate_returns(self, sequence_length: int, outcome: int) -> List[float]:
        """Calculate returns for each step in the sequence."""
        gamma = 0.95  # Higher discount factor for longer-term planning
        returns = []
        current_return = float(outcome)
        
        for _ in range(sequence_length):
            returns.append(current_return)
            current_return *= gamma
        
        returns.reverse()  # So index 0 is for first move
        return returns
    
    def _extract_patterns(self, state: np.ndarray, player: int) -> List[str]:
        """Extract 8 canonical patterns (rows, cols, diags) from player's perspective, using caching."""
        # Create a hashable cache key: (player, board_bytes)
        cache_key = (player, state.tobytes())
        
        if cache_key in self._pattern_cache:
            self._cache_hits += 1
            return self._pattern_cache[cache_key]
        else:
            self._cache_misses += 1
            patterns = []
            # Convert board to player's perspective (-1 opponent, 1 player, 0 empty)
            perspective_state = state.copy()
            perspective_state[perspective_state == player] = 1
            perspective_state[perspective_state == -player] = -1
            perspective_state[perspective_state == 0] = 0
            
            # Convert to pattern string format (1 player, 2 opponent, 0 empty)
            pattern_state = perspective_state.copy()
            pattern_state[pattern_state == -1] = 2
            pattern_state[pattern_state == 1] = 1
            pattern_state[pattern_state == 0] = 0
            
            # 1. Extract row patterns (3 patterns)
            for i, row in enumerate(pattern_state):
                pattern = "".join(map(str, row.astype(int)))
                patterns.append(f"row_{pattern}")
                
            # 2. Extract column patterns (3 patterns)
            for i, col in enumerate(pattern_state.T):
                pattern = "".join(map(str, col.astype(int)))
                patterns.append(f"col_{pattern}")
                
            # 3. Extract diagonal patterns (2 patterns)
            diag = np.diagonal(pattern_state)
            anti_diag = np.diagonal(np.fliplr(pattern_state))
            
            diag_str = "".join(map(str, diag.astype(int)))
            anti_diag_str = "".join(map(str, anti_diag.astype(int)))
            patterns.append(f"diag_{diag_str}")
            patterns.append(f"anti_diag_{anti_diag_str}")
            
            # No need to generate mirrors or rotations here
            # Equivalency is handled by StrategicMemory.get_pattern_value
            
            # Store in cache before returning (should be exactly 8 patterns now)
            self._pattern_cache[cache_key] = patterns
            return patterns
    
    def _get_state_key(self, state):
        """Convert a board state to a string key."""
        return "_".join(map(str, state.flatten())) 