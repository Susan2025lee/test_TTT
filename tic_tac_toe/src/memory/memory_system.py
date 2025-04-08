from typing import List, Tuple, Dict, Optional
import numpy as np
from ..environment.board import Board
from .short_term import ShortTermMemory
from .experience import ExperienceMemory
from .strategic import StrategicMemory

class MemorySystem:
    """
    Memory system manager that coordinates all memory components
    and provides a unified interface for the agent.
    """
    
    def __init__(self):
        """Initialize the memory system."""
        self.short_term = ShortTermMemory()
        self.experience = ExperienceMemory()
        self.strategic = StrategicMemory()
        
    def add_state(self, board: Board, move: Optional[Tuple[int, int]] = None):
        """
        Add a new state to short-term memory.
        
        Args:
            board: Current game board
            move: The move that led to this state (if any)
        """
        # Calculate move score if move was made
        score = 0.0
        if move is not None:
            score = self.strategic.get_strategic_value(move, board.get_state())
            
        self.short_term.add_state(board, move, score)
        
    def end_game(self, outcome: int):
        """
        Process the end of a game.
        
        Args:
            outcome: Game outcome (1: win, 0: draw, -1: loss)
        """
        # Get game history
        states = self.short_term.get_state_history()
        moves = self.short_term.get_current_sequence()
        
        # Update experience memory
        self.experience.add_experience(
            states=states,
            moves=moves,
            outcome=outcome,
            player_sequence=[1, -1] * len(moves)  # Alternating players
        )
        
        # Update strategic memory
        self.strategic.add_game(states, moves, outcome)
        
        # Reset short-term memory
        self.short_term.reset()
        
    def get_move_recommendation(self, 
                              board: Board,
                              n_recommendations: int = 3) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get move recommendations based on all memory components.
        
        Args:
            board: Current game board
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (move, confidence) tuples
        """
        state = board.get_state()
        valid_moves = board.get_valid_moves()
        recommendations = []
        
        # Get strategic values for all valid moves
        for move in valid_moves:
            value = self.strategic.get_strategic_value(move, state)
            
            # Add bonus for winning moves
            if move in self.strategic.get_winning_moves(state):
                value += 0.5
                
            # Add bonus for successful counter-moves
            last_move = self.short_term.current_sequence[-1] if self.short_term.current_sequence else None
            if last_move:
                counters = self.strategic.get_best_counter(last_move)
                for counter_move, counter_rate in counters:
                    if counter_move == move:
                        value += 0.3 * counter_rate
                        
            # Add bonus for moves that worked in similar states
            similar_states = self.experience.get_similar_states(state)
            for similar in similar_states:
                if similar['move'] == move:
                    value += 0.2 * (1 if similar['outcome'] == 1 else 0.5)
                    
            recommendations.append((move, value))
            
        # Sort by value and return top n
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:n_recommendations]
        
    def get_context(self) -> Dict:
        """
        Get the current game context from all memory components.
        
        Returns:
            Dict containing context information from all memory systems
        """
        return {
            'short_term': self.short_term.get_context(),
            'best_moves': self.experience.get_best_moves(),
            'position_values': self.strategic.position_values.copy()
        }
        
    def clear(self):
        """Clear all memory components."""
        self.short_term.reset()
        self.experience.clear()
        self.strategic.clear() 