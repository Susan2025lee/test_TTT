"""Tests for the training system."""

import pytest
import numpy as np
from tic_tac_toe.src.agent.core import Agent
from tic_tac_toe.src.memory.memory_system import MemorySystem
from tic_tac_toe.src.reward.reward_system import RewardSystem
from tic_tac_toe.src.learning.state_eval import StateEvaluator
from tic_tac_toe.src.learning.action_selection import ActionSelector
from tic_tac_toe.src.learning.experience_processor import ExperienceProcessor
from tic_tac_toe.src.training.self_play import SelfPlayTrainer

@pytest.fixture
def agent():
    """Create a test agent with all components."""
    memory_system = MemorySystem()
    reward_system = RewardSystem()
    state_evaluator = StateEvaluator()
    action_selector = ActionSelector(state_evaluator=state_evaluator)
    experience_processor = ExperienceProcessor(
        experience_memory=memory_system.experience,
        strategic_memory=memory_system.strategic,
        state_evaluator=state_evaluator
    )
    return Agent(
        memory_system=memory_system,
        reward_system=reward_system,
        state_evaluator=state_evaluator,
        action_selector=action_selector,
        experience_processor=experience_processor,
        player_id=1
    )

def test_trainer_initialization(agent):
    """Test that trainer initializes correctly."""
    trainer = SelfPlayTrainer(agent, num_games=100)
    assert trainer.agent1 is agent
    assert trainer.agent2 is agent
    assert trainer.num_games == 100
    assert trainer.eval_frequency == 100
    assert isinstance(trainer.metrics, dict)
    
def test_single_game_training(agent):
    """Test that trainer can run a single game."""
    trainer = SelfPlayTrainer(agent, num_games=1)
    metrics = trainer.train()
    
    assert isinstance(metrics, dict)
    assert len(metrics['win_rates']) > 0
    assert len(metrics['draw_rates']) > 0
    assert len(metrics['avg_game_length']) > 0
    
def test_training_metrics(agent):
    """Test that training produces valid metrics."""
    trainer = SelfPlayTrainer(agent, num_games=10, eval_frequency=5)
    metrics = trainer.train()
    
    # Check metric shapes
    assert len(metrics['win_rates']) == 2  # 10 games / 5 frequency = 2 evaluations
    assert len(metrics['draw_rates']) == 2
    assert len(metrics['avg_game_length']) == 2
    assert len(metrics['learning_curves']) == 2
    
    # Check metric values
    for rate in metrics['win_rates']:
        assert 0 <= rate <= 1
    for rate in metrics['draw_rates']:
        assert 0 <= rate <= 1
    for length in metrics['avg_game_length']:
        assert 0 < length <= 9  # Max 9 moves in Tic Tac Toe
        
def test_two_agent_training(agent):
    """Test training with two different agents."""
    # Create second agent
    memory_system = MemorySystem()
    reward_system = RewardSystem()
    state_evaluator = StateEvaluator()
    action_selector = ActionSelector(state_evaluator=state_evaluator)
    experience_processor = ExperienceProcessor(
        experience_memory=memory_system.experience,
        strategic_memory=memory_system.strategic,
        state_evaluator=state_evaluator
    )
    agent2 = Agent(
        memory_system=memory_system,
        reward_system=reward_system,
        state_evaluator=state_evaluator,
        action_selector=action_selector,
        experience_processor=experience_processor,
        player_id=-1
    )
    
    trainer = SelfPlayTrainer(agent, agent2, num_games=10)
    metrics = trainer.train()
    
    assert isinstance(metrics, dict)
    assert len(metrics['win_rates']) > 0
    
def test_game_outcome_processing(agent):
    """Test that game outcomes are processed correctly."""
    trainer = SelfPlayTrainer(agent, num_games=5)
    
    # Play a few games and check outcomes
    for _ in range(3):
        trainer.game_env.reset()
        
        while not trainer.game_env.is_game_over():
            current_agent = trainer.agent1 if trainer.game_env.get_current_player() == 1 else trainer.agent2
            move = current_agent.select_move(trainer.game_env.get_board())
            if move is None:
                break
            trainer.game_env.make_move(move)
            
        outcome = trainer._get_game_outcome()
        assert outcome in [-1, 0, 1]  # Valid game outcomes 