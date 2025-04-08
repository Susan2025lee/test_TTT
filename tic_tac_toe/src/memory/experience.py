import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import time

class ExperienceMemory:
    """
    Experience memory component that stores and manages past game experiences,
    including complete games, outcomes, and patterns.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize experience memory.
        
        Args:
            capacity: Maximum number of games to store
        """
        self.capacity = capacity
        self.experiences = []  # List of game experiences
        self.pattern_index = defaultdict(list)  # Index of board patterns
        self.move_stats = defaultdict(lambda: {'wins': 0, 'total': 0})  # Move statistics
        
    def add_experience(self, 
                      states: List[np.ndarray],
                      moves: List[Tuple[int, int]],
                      outcome: int,
                      player_sequence: List[int]):
        """
        Add a complete game experience to memory.
        
        Args:
            states: List of board states
            moves: List of moves made
            outcome: Game outcome (1: win, 0: draw, -1: loss)
            player_sequence: Sequence of players for each move
        """
        experience = {
            'states': states,
            'moves': moves,
            'outcome': outcome,
            'players': player_sequence,
            'timestamp': time.time()
        }
        
        # Add experience
        self.experiences.append(experience)
        
        # Update pattern index
        self._update_pattern_index(experience)
        
        # Update move statistics
        for move, player in zip(moves, player_sequence):
            self.move_stats[move]['total'] += 1
            if outcome == 1 and player == 1:  # Win for player 1
                self.move_stats[move]['wins'] += 1
            elif outcome == -1 and player == -1:  # Win for player -1
                self.move_stats[move]['wins'] += 1
        
        # Maintain capacity
        if len(self.experiences) > self.capacity:
            oldest = self.experiences.pop(0)
            self._remove_from_pattern_index(oldest)
            
    def store_game(self, states: List[np.ndarray], moves: List[Tuple[int, int]], outcome: int):
        """
        Store a complete game in memory.
        
        Args:
            states: List of board states
            moves: List of moves made
            outcome: Game outcome (1: win, 0: draw, -1: loss)
        """
        # Determine player sequence based on move order (X starts as 1, O as -1)
        player_sequence = [1 if i % 2 == 0 else -1 for i in range(len(moves))]
        
        # Add the experience with the inferred player sequence
        self.add_experience(states, moves, outcome, player_sequence)
        
    def _update_pattern_index(self, experience: Dict):
        """Update the pattern index with new experience."""
        for state, move in zip(experience['states'], experience['moves']):
            pattern_key = self._create_pattern_key(state)
            self.pattern_index[pattern_key].append({
                'move': move,
                'outcome': experience['outcome'],
                'timestamp': experience['timestamp']
            })
            
    def _remove_from_pattern_index(self, experience: Dict):
        """Remove an experience from the pattern index."""
        for state in experience['states']:
            pattern_key = self._create_pattern_key(state)
            if pattern_key in self.pattern_index:
                self.pattern_index[pattern_key] = [
                    entry for entry in self.pattern_index[pattern_key]
                    if entry['timestamp'] != experience['timestamp']
                ]
                
    def _create_pattern_key(self, state: np.ndarray) -> str:
        """Create a string key for a board pattern."""
        return ','.join(map(str, state.flatten()))
        
    def get_similar_states(self, state: np.ndarray, n: int = 5) -> List[Dict]:
        """
        Get similar states from memory.
        
        Args:
            state: Current board state
            n: Number of similar states to return
            
        Returns:
            List of similar states with their moves and outcomes
        """
        pattern_key = self._create_pattern_key(state)
        similar_states = self.pattern_index.get(pattern_key, [])
        
        # Sort by recency and outcome
        sorted_states = sorted(
            similar_states,
            key=lambda x: (x['outcome'], x['timestamp']),
            reverse=True
        )
        
        return sorted_states[:n]
        
    def get_move_stats(self, move: Tuple[int, int]) -> Dict:
        """Get statistics for a specific move."""
        stats = self.move_stats[move]
        return {
            'total': stats['total'],
            'wins': stats['wins'],
            'win_rate': stats['wins'] / stats['total'] if stats['total'] > 0 else 0
        }
        
    def get_best_moves(self, n: int = 3) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get the n best moves based on win rate.
        
        Returns:
            List of (move, win_rate) tuples
        """
        move_rates = [
            (move, stats['wins'] / stats['total'])
            for move, stats in self.move_stats.items()
            if stats['total'] > 0
        ]
        
        return sorted(move_rates, key=lambda x: x[1], reverse=True)[:n]
        
    def clear(self):
        """Clear all experiences."""
        self.experiences = []
        self.pattern_index.clear()
        self.move_stats.clear() 