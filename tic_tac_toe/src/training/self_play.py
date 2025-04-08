"""Self-play training module for the Tic Tac Toe agent.

This module implements self-play training for the Tic Tac Toe agent, allowing it to learn
and improve through playing against itself or another agent. The training process includes:
- Game simulation with move selection and execution
- Experience collection and processing
- Performance evaluation and metrics tracking
- Learning through game outcomes
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from ..agent import Agent
from ..environment.board import Board
from ..environment.game_env import GameEnvironment

class SelfPlayTrainer:
    """Handles self-play training for the Tic Tac Toe agent.
    
    The trainer manages the training process by:
    1. Running games between agents
    2. Collecting game experiences
    3. Processing outcomes
    4. Updating agent knowledge
    5. Tracking performance metrics
    
    The training can be configured for:
    - Single agent self-play (agent plays against itself)
    - Two agent training (agents compete and learn from each other)
    - Custom evaluation frequency
    - Variable number of training games
    
    Attributes:
        agent1 (Agent): Primary agent being trained
        agent2 (Agent): Opponent agent (same as agent1 for self-play)
        num_games (int): Total number of games to play
        eval_frequency (int): How often to evaluate performance
        game_env (GameEnvironment): Game environment for simulating matches
        metrics (Dict): Dictionary tracking various performance metrics
    """
    
    def __init__(self,
                 agent1: Agent,
                 agent2: Optional[Agent] = None,
                 num_games: int = 1000,
                 eval_frequency: int = 100):
        """Initialize the self-play trainer.
        
        Args:
            agent1: First agent (will be trained)
            agent2: Second agent (optional, if None will use copy of agent1)
            num_games: Number of games to play in training
            eval_frequency: How often to evaluate performance (in games)
        """
        self.agent1 = agent1
        self.agent2 = agent2 if agent2 else agent1
        self.num_games = num_games
        self.eval_frequency = eval_frequency
        self.game_env = GameEnvironment()
        
        # Training metrics
        self.metrics = {
            'win_rates': [],      # Win rate during evaluation
            'draw_rates': [],     # Draw rate during evaluation
            'avg_game_length': [], # Average game length during evaluation
            'learning_curves': []  # Combined metrics over time
        }
        
    def train(self) -> Dict:
        """Run the training loop.
        
        The training process consists of:
        1. Playing games between agents
        2. Collecting game experiences
        3. Processing outcomes and updating agents
        4. Periodic performance evaluation
        
        Returns:
            Dict containing training metrics:
            - win_rates: List of win rates during evaluation
            - draw_rates: List of draw rates during evaluation
            - avg_game_length: List of average game lengths
            - learning_curves: List of combined metrics at each evaluation
        """
        for game_num in range(self.num_games):
            # Reset game environment
            self.game_env.reset()
            game_history = []
            
            # Play a game
            while not self.game_env.is_game_over():
                current_agent = self.agent1 if self.game_env.get_current_player() == 1 else self.agent2
                board_state = self.game_env.get_board()
                
                # Select and make move
                move = current_agent.select_move(board_state)
                if move is None:
                    break
                    
                self.game_env.make_move(move)
                game_history.append((board_state.copy(), move))
                
            # Process game outcome
            outcome = self._get_game_outcome()
            self._update_agents(game_history, outcome)
            
            # Evaluate performance periodically
            if (game_num + 1) % self.eval_frequency == 0:
                self._evaluate_performance(game_num + 1)
                
        # Ensure at least one evaluation at the end
        if not self.metrics['win_rates']:
            self._evaluate_performance(self.num_games)
                
        return self.metrics
        
    def _get_game_outcome(self) -> int:
        """Get the game outcome from agent1's perspective.
        
        Returns:
            int: 1 for win, -1 for loss, 0 for draw
        """
        winner = self.game_env.get_winner()
        if winner is None:
            return 0  # Draw
        return 1 if winner == 1 else -1
        
    def _update_agents(self, 
                      game_history: List[Tuple[Board, Tuple[int, int]]], 
                      outcome: int):
        """Update both agents with the game experience.
        
        This method processes the game outcome and updates the agents'
        knowledge based on their performance. For two different agents,
        the outcome is inverted for the second agent.
        
        Args:
            game_history: List of (board_state, move) tuples from the game
            outcome: Game result (1 for win, -1 for loss, 0 for draw)
        """
        # Update agent1
        self.agent1.update_from_game_result(outcome)
        
        # Update agent2 if it's different from agent1
        if self.agent2 is not self.agent1:
            self.agent2.update_from_game_result(-outcome)  # Opposite outcome
            
    def _evaluate_performance(self, game_num: int):
        """Evaluate current agent performance.
        
        Runs a series of evaluation games to assess the current
        performance level of the agents. Updates various metrics
        including win rates, draw rates, and game lengths.
        
        Args:
            game_num: Current game number in training
        """
        # Play evaluation games
        wins = draws = total_length = 0
        eval_games = 100
        
        for _ in range(eval_games):
            self.game_env.reset()
            moves = 0
            
            while not self.game_env.is_game_over():
                current_agent = self.agent1 if self.game_env.get_current_player() == 1 else self.agent2
                move = current_agent.select_move(self.game_env.get_board())
                if move is None:
                    break
                self.game_env.make_move(move)
                moves += 1
                
            outcome = self._get_game_outcome()
            if outcome == 1:
                wins += 1
            elif outcome == 0:
                draws += 1
            total_length += moves
            
        # Update metrics
        self.metrics['win_rates'].append(wins / eval_games)
        self.metrics['draw_rates'].append(draws / eval_games)
        self.metrics['avg_game_length'].append(total_length / eval_games)
        self.metrics['learning_curves'].append({
            'game_num': game_num,
            'win_rate': wins / eval_games,
            'draw_rate': draws / eval_games,
            'avg_length': total_length / eval_games
        }) 