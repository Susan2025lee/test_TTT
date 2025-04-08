"""State evaluation module for the Tic Tac Toe agent."""

from typing import Dict, List, Tuple
import numpy as np

from ..environment import Board

class StateEvaluator:
    """Evaluates board states and extracts relevant features."""
    
    def __init__(self):
        """Initialize the state evaluator with predefined patterns."""
        # Winning patterns (rows, columns, diagonals)
        self.win_patterns = [
            [(i, j) for j in range(3)] for i in range(3)  # Rows
        ] + [
            [(i, j) for i in range(3)] for j in range(3)  # Columns
        ] + [
            [(i, i) for i in range(3)],  # Main diagonal
            [(i, 2-i) for i in range(3)]  # Anti-diagonal
        ]
        
        # Strategic positions
        self.center = (1, 1)
        self.corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        self.edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        
        # Common game patterns
        self.patterns = {
            'corner_domination': [
                [(0, 0), (0, 2)],  # Top corners
                [(2, 0), (2, 2)],  # Bottom corners
                [(0, 0), (2, 0)],  # Left corners
                [(0, 2), (2, 2)]   # Right corners
            ],
            'triangle_pattern': [
                [(0, 0), (1, 1), (2, 2)],  # Main diagonal triangle
                [(0, 2), (1, 1), (2, 0)],  # Anti-diagonal triangle
                [(0, 0), (1, 1), (0, 2)],  # Top triangle
                [(2, 0), (1, 1), (2, 2)]   # Bottom triangle
            ],
            'l_pattern': [
                [(0, 0), (0, 1), (1, 0)],  # Top-left L
                [(0, 1), (0, 2), (1, 2)],  # Top-right L
                [(1, 0), (2, 0), (2, 1)],  # Bottom-left L
                [(1, 2), (2, 1), (2, 2)]   # Bottom-right L
            ]
        }
        
        # Feature weights for strategic evaluation
        self.weights = {
            # Position control weights
            'center_control': 0.15,
            'corner_control': 0.1,
            'edge_control': 0.05,
            
            # Line control weights (for each line)
            'line_potential': 0.15,  # Increased from 0.1
            'line_blocked': -0.1,  # Increased penalty from -0.05
            
            # Pattern weights
            'corner_domination': 0.15,
            'triangle_control': 0.1,
            'l_pattern_control': 0.1,
            'pattern_strength': 0.2,  # Increased from 0.15
            
            # Threat weights
            'winning_moves': 0.5,  # Increased from 0.3
            'blocking_moves': 0.4,  # Increased from 0.2
            
            # Mobility weight
            'mobility': 0.1
        }
        
    def evaluate_state(self, board: Board) -> float:
        """
        Evaluate the current board state from the perspective of the current player.
        
        Args:
            board: Current game board
            
        Returns:
            Float value between -1 and 1 indicating the state value
        """
        current_player = board.current_player
        winner = board.check_winner() # Check terminal state first
        if winner == current_player:
            return 1.0
        elif winner == -current_player:
            return -1.0
        elif board.is_draw():
             return 0.0
        
        # Check for immediate win/loss threats (for evaluation, not early return)
        # winning_threat = False
        # blocking_threat = False
        # for pattern in self.win_patterns:
        #     values = [board.board[r, c] for r, c in pattern]
        #     if values.count(current_player) == 2 and values.count(0) == 1:
        #         winning_threat = True
        #     if values.count(-current_player) == 2 and values.count(0) == 1:
        #         blocking_threat = True
        # Removed premature returns based on threats
        
        # Extract all features
        features = self.extract_features(board)
        
        # Initialize evaluation components
        position_score = (
            features['center_control'] * self.weights['center_control'] +
            features['corner_control'] * self.weights['corner_control'] +
            features['edge_control'] * self.weights['edge_control']
        )
        
        # Calculate line control score with higher weight for potential wins
        line_score = 0.0
        for i in range(8):  # 8 possible lines
            line_score += (
                features.get(f'line_{i}_potential', 0.0) * self.weights['line_potential'] * 1.5 +  # 50% boost
                features.get(f'line_{i}_blocked', 0.0) * self.weights['line_blocked']
            )
        
        # Calculate pattern score
        pattern_score = (
            features['corner_domination'] * self.weights['corner_domination'] +
            features['triangle_control'] * self.weights['triangle_control'] +
            features['l_pattern_control'] * self.weights['l_pattern_control'] +
            features['pattern_strength'] * self.weights['pattern_strength']
        )
        
        # Calculate threat score with higher weight for winning moves
        threat_score = (
            features['winning_moves'] * (self.weights['winning_moves'] * 2.0) +  # Double weight
            features['blocking_moves'] * (self.weights['blocking_moves'] * 2.0)  # Equal weight to winning
        )
        
        # Add mobility score
        mobility_score = features['mobility'] * self.weights['mobility']
        
        # Combine all components with adjusted weights
        total_score = (
            position_score * 0.15 +
            line_score * 0.25 +
            pattern_score * 0.2 +
            threat_score * 0.35 +
            mobility_score * 0.05
        )
        
        # Normalize to [-1, 1] range using tanh
        return np.tanh(total_score)
        
    def extract_features(self, board: Board) -> Dict[str, float]:
        """
        Extract relevant features from the current board state.
        
        Args:
            board: Current game board
            
        Returns:
            Dictionary of features and their values
        """
        features = {}
        state = board.get_state()
        current_player = board.current_player

        # Handle empty board case first
        if np.all(state == 0):
            # For empty board, all strategic features are 0, mobility is max
            # Initialize features expected by evaluate_state to 0.0
            features['center_control'] = 0.0
            features['corner_control'] = 0.0
            features['edge_control'] = 0.0
            for i in range(8): 
                features[f'line_{i}_potential'] = 0.0
                features[f'line_{i}_blocked'] = 0.0
            features['corner_domination'] = 0.0
            features['triangle_control'] = 0.0
            features['l_pattern_control'] = 0.0
            features['pattern_strength'] = 0.0
            features['winning_moves'] = 0.0
            features['blocking_moves'] = 0.0
            features['mobility'] = 1.0 # Max mobility (9/9)
            return features
        
        # Basic position features (calculated relative to current player)
        features['center_control'] = state[self.center] * current_player if state[self.center] != 0 else 0.0
        features['corner_control'] = np.mean([state[c] * current_player for c in self.corners if state[c] != 0] + [0.0])
        features['edge_control'] = np.mean([state[e] * current_player for e in self.edges if state[e] != 0] + [0.0])
        
        # Line control features
        features.update(self._evaluate_line_control(board))
        
        # Pattern recognition features
        pattern_features = self._recognize_patterns(board)
        features.update(pattern_features)
        
        # Threat features
        threats = self.detect_threats(board)
        features['winning_moves'] = len(threats['winning_moves'])
        features['blocking_moves'] = len(threats['blocking_moves'])
        
        # Mobility features (needs to be relative to total possible moves)
        features['mobility'] = len(board.get_valid_moves()) / 9.0
        
        return features
    
    def _recognize_patterns(self, board: Board) -> Dict[str, float]:
        """
        Recognize common game patterns on the board.
        
        Args:
            board: Current game board
            
        Returns:
            Dictionary of pattern features and their values
        """
        features = {}
        state = board.get_state()
        current_player = board.current_player
        
        # Check corner domination
        corner_count = 0
        total_corners = len(self.corners)
        for corner in self.corners:
            if state[corner] == 1:  # X has the corner
                corner_count += 1
            elif state[corner] == -1:  # O has the corner
                corner_count -= 1
        features['corner_domination'] = corner_count / total_corners
        
        # Check triangle patterns
        triangle_control = 0
        for triangle in self.patterns['triangle_pattern']:
            values = [state[r, c] for r, c in triangle]
            # Check for both current player and opponent's pieces
            if values.count(1) >= 2 and values.count(-1) == 0:  # X's triangle
                triangle_control += 1
            elif values.count(-1) >= 2 and values.count(1) == 0:  # O's triangle
                triangle_control -= 1
        features['triangle_control'] = triangle_control / len(self.patterns['triangle_pattern'])
        
        # Check L patterns
        l_pattern_control = 0
        for l_pattern in self.patterns['l_pattern']:
            values = [state[r, c] for r, c in l_pattern]
            # Check for both current player and opponent's pieces
            if values.count(1) >= 2 and values.count(-1) == 0:  # X's L pattern
                l_pattern_control += 1
            elif values.count(-1) >= 2 and values.count(1) == 0:  # O's L pattern
                l_pattern_control -= 1
        features['l_pattern_control'] = l_pattern_control / len(self.patterns['l_pattern'])
        
        # Calculate pattern strength based on who has control
        x_control = (
            max(0, features['corner_domination']) * 0.4 +
            max(0, features['triangle_control']) * 0.3 +
            max(0, features['l_pattern_control']) * 0.3
        )
        
        o_control = (
            max(0, -features['corner_domination']) * 0.4 +
            max(0, -features['triangle_control']) * 0.3 +
            max(0, -features['l_pattern_control']) * 0.3
        )
        
        # Pattern strength is positive if current player has more control
        if current_player == 1:  # X's turn
            features['pattern_strength'] = x_control - o_control
        else:  # O's turn
            features['pattern_strength'] = o_control - x_control
        
        return features
        
    def assess_position(self, board: Board, player_id: int) -> Dict[str, float]:
        """Assess positional features (center, corner, edge control) 
           from the perspective of the given player_id.
           Does NOT simulate moves.
        """
        state = board.get_state()
        features = {
            'center_control': 0.0,
            'corner_control': 0.0,
            'edge_control': 0.0
        }

        # Center control
        center_owner = state[self.center]
        # Ensure comparison is with scalar value if center_owner is numpy array/object
        if isinstance(center_owner, np.ndarray):
            center_owner = center_owner.item() # Extract scalar value
        if center_owner == player_id:
            features['center_control'] = 1.0
        elif center_owner == -player_id:
            features['center_control'] = -1.0

        # Corner control (average)
        corner_values = []
        for corner in self.corners:
            owner = state[corner]
            if isinstance(owner, np.ndarray):
                 owner = owner.item()
            if owner == player_id:
                corner_values.append(1.0)
            elif owner == -player_id:
                corner_values.append(-1.0)
        features['corner_control'] = np.nan_to_num(np.mean(corner_values + [0.0]))

        # Edge control (average)
        edge_values = []
        for edge in self.edges:
            owner = state[edge]
            if isinstance(owner, np.ndarray):
                 owner = owner.item()
            if owner == player_id:
                edge_values.append(1.0)
            elif owner == -player_id:
                edge_values.append(-1.0)
        features['edge_control'] = np.nan_to_num(np.mean(edge_values + [0.0]))
        
        return features
        
    def detect_threats(self, board: Board) -> Dict[str, List[Tuple[int, int]]]:
        """
        Detect immediate winning moves (threats) for both players.
        Simpler implementation: iterate valid moves and check for win.
        """
        threats = {'winning_moves': [], 'blocking_moves': []}
        current_player = board.current_player
        valid_moves = board.get_valid_moves()

        # --- Debug for test_threat_detection Scenario 1 ---
        # Use get_state()
        # debug_state_check = np.array_equal(board.get_state(), np.array([[1,-1,0],[0,1,0],[-1,0,0]])) and board.current_player == 1
        # if debug_state_check:
        #     print("\nDEBUG: detect_threats running for test_state_1")
        #     print(f"Valid moves: {valid_moves}")
        # --- End Debug ---

        # print(f"\nDEBUG detect_threats: Checking threats for Player {current_player}")
        # print(f"Board state:\n{board.get_state()}")
        # print(f"Valid moves: {valid_moves}")

        for move in valid_moves:
            # print(f"  Checking move: {move}")
            # Check if move wins for current player
            board_after_move = board.copy()
            # print(f"    Board state before make_move({move}):\\n{board_after_move.get_state()}")
            # print(f"    Player before make_move({move}): {board_after_move.current_player}")
            board_after_move.make_move(move) # state updated, player switched
            winner = board_after_move.check_winner() # Should return ID of winner (1, -1, or 0)
            # print(f"    Board after move {move} (player is now {board_after_move.current_player}):")
            # print(f"    Winner after move {move}: {winner} (type: {type(winner)})")

            move_wins = (winner == current_player) # Check if the winner is the player whose turn it was
            # print(f"    Does this move win for Player {current_player}? {move_wins} (Winner: {winner}, Current Player: {current_player})")

            if move_wins:
                threats['winning_moves'].append(move)
                # print(f"    !!! Added {move} to winning_moves")

            # Check if move would win for opponent (needs blocking)
            # Avoid redundant check if the move already won for the current player
            if not move_wins:
                board_opponent_move = board.copy()
                # Temporarily switch perspective to opponent for the check
                original_opponent_perspective_player = -current_player
                board_opponent_move.current_player = original_opponent_perspective_player
                # print(f"    Checking if move {move} blocks opponent (Player {original_opponent_perspective_player})")
                # print(f"      Board state before opponent make_move({move}):\\n{board_opponent_move.get_state()}")
                board_opponent_move.make_move(move) # Simulate opponent making the move
                opponent_winner = board_opponent_move.check_winner() # Check if opponent wins
                # print(f"      Board after opponent move {move} (player is now {board_opponent_move.current_player}):\\n{board_opponent_move.get_state()}")
                # print(f"      Winner if opponent played {move}: {opponent_winner}")

                # Check if the opponent *would* win if they made that move
                if opponent_winner == original_opponent_perspective_player:
                    threats['blocking_moves'].append(move)
                    # print(f"    !!! Added {move} to blocking_moves")

        # print(f"DEBUG detect_threats: Result = {threats}\n")
        return threats
        
    def _evaluate_position_control(self, board: Board, pos: Tuple[int, int]) -> float:
        """Evaluate control of a specific position."""
        value = board.board[pos]
        if value == 0:
            return 0.0
        return 1.0 if value == 1 else -1.0  # 1 for X (first player), -1 for O
        
    def _evaluate_line_control(self, board: Board) -> Dict[str, float]:
        """
        Evaluate control over each of the 8 potential winning lines.
        Returns features like line_X_potential, line_X_blocked.
        """
        line_features = {}
        current_player = board.current_player

        for i, pattern_coords in enumerate(self.win_patterns):
            values = [board.board[r, c] for r, c in pattern_coords]
            player_count = values.count(current_player)
            opponent_count = values.count(-current_player)
            empty_count = values.count(0)

            potential = 0.0
            blocked = 0.0

            if opponent_count > 0: # If opponent is in the line, player cannot win on it
                potential = 0.0 
                if player_count > 0: # If player is also in the line, it's blocked/contested
                    blocked = 0.5 # Partial block
                else: # Opponent has potential control
                    blocked = 0.1 # Minimal block value, mainly opponent potential
            elif player_count == 2 and empty_count == 1:
                potential = 0.9 # High potential (immediate threat)
            elif player_count == 1 and empty_count == 2:
                potential = 0.4 # Moderate potential
            elif empty_count == 3:
                potential = 0.1 # Low potential (empty line)
            elif player_count == 3:
                 potential = 1.0 # Already won (should be caught by evaluate_state)
            
            # Store relative to the current player
            line_features[f'line_{i}_potential'] = potential
            # Blocked is negative, assign based on opponent presence
            line_features[f'line_{i}_blocked'] = -1.0 if opponent_count > 0 else 0.0 
            
        return line_features

    def _assess_line_potential(self, board: Board, move: Tuple[int, int]) -> float:
        """
        Assess the potential value created by placing a piece at 'move' 
        in terms of completing lines.
        """
        potential = 0.0
        current_player = board.current_player
        count = 0

        # Simulate placing the piece (assume move is valid)
        state_after = board.board.copy()
        state_after[move] = current_player

        for pattern_coords in self.win_patterns:
            # Only check lines involving the current move
            if move in pattern_coords:
                count += 1
                values = [state_after[r, c] for r, c in pattern_coords]
                player_count = values.count(current_player)
                opponent_count = values.count(-current_player)
                empty_count = values.count(0) # Should be 0 after placing piece

                if opponent_count == 0: # No opponent blocking this line
                    if player_count == 3:
                        potential += 1.0 # Creates a win
                    elif player_count == 2:
                         # This move doesn't create 2-in-a-row, it means *before* the move
                         # there was 1 player piece and 2 empty in the line.
                         potential += 0.4 # Moderate potential created
                    elif player_count == 1:
                         # Means the line was empty before this move.
                         potential += 0.1 # Low potential created
        
        # Average potential over lines affected by the move
        return potential / count if count > 0 else 0.0
        
    def _assess_pattern_value(self, board: Board, move: Tuple[int, int]) -> float:
        """
        Assess how well a move completes or contributes to strategic patterns.
        
        Args:
            board: Current game board
            move: Position to evaluate
            
        Returns:
            Pattern completion value between 0 and 1
        """
        value = 0.0
        board_copy = board.copy()
        board_copy.make_move(move)
        state = board_copy.get_state()
        player = board.current_player
        
        # Check if move completes corner domination
        for corner_pair in self.patterns['corner_domination']:
            if move in corner_pair:
                values = [state[r, c] for r, c in corner_pair]
                if values.count(player) == 2:
                    value += 0.4
                    
        # Check if move contributes to triangle pattern
        for triangle in self.patterns['triangle_pattern']:
            if move in triangle:
                values = [state[r, c] for r, c in triangle]
                if values.count(player) >= 2 and values.count(-player) == 0:
                    value += 0.3
                    
        # Check if move completes L pattern
        for l_pattern in self.patterns['l_pattern']:
            if move in l_pattern:
                values = [state[r, c] for r, c in l_pattern]
                if values.count(player) >= 2 and values.count(-player) == 0:
                    value += 0.3
                    
        return min(value, 1.0)  # Cap at 1.0 

    def _creates_fork(self, board: Board, move: Tuple[int, int], player_id: int) -> bool:
        """Check if placing player_id at move creates a fork.

        A fork is created if the move results in at least two lines 
        each having two of the player's pieces and one empty space.
        """
        # Ensure the move is valid before proceeding
        if not board.is_valid_move(move):
            return False

        # Simulate the move on a copy
        temp_board = board.copy()
        temp_board.make_move(move)

        # If the move wins, it's not considered creating a fork
        # Need to check against the player_id who made the move
        if temp_board.check_winner() == player_id:
             # Note: make_move flips the current player in the copy.
             # We are checking if the simulated move by player_id resulted in a win FOR player_id.
             # However, check_winner just returns the winner ID (1 or -1). 
             # We need to compare with the original player_id passed to this function.
             # Let's adjust the check:
             pass # The check below implicitly handles this, as a winning line has 3 pieces.

        # Get state *after* the move
        state_after = temp_board.get_state()
        row, col = move
        threats_created = 0

        # Check Row containing the move
        the_row = state_after[row, :]
        if np.count_nonzero(the_row == player_id) == 2 and np.count_nonzero(the_row == 0) == 1:
            threats_created += 1

        # Check Column containing the move
        the_col = state_after[:, col]
        if np.count_nonzero(the_col == player_id) == 2 and np.count_nonzero(the_col == 0) == 1:
            threats_created += 1

        # Check Main Diagonal (if move is on it)
        if row == col:
            the_diag = np.diag(state_after)
            if np.count_nonzero(the_diag == player_id) == 2 and np.count_nonzero(the_diag == 0) == 1:
                threats_created += 1

        # Check Anti-Diagonal (if move is on it)
        if row + col == 2:
            # Need to prevent double-counting if move is (1,1) which is on both diagonals
            if row != col: # Only check anti-diag separately if not the center square
                the_anti_diag = np.diag(np.fliplr(state_after))
                if np.count_nonzero(the_anti_diag == player_id) == 2 and np.count_nonzero(the_anti_diag == 0) == 1:
                    threats_created += 1
            # If move is (1,1), the main diagonal check already covered it.

        # --- Debug for test_fork_detection Scenario 1 --- 
        debug_fork_check = np.array_equal(board.get_state(), np.array([[1,0,0],[0,-1,0],[0,0,1]])) and player_id == 1 and move == (0,2)
        if debug_fork_check:
            print("\nDEBUG: _creates_fork running for test_state_1")
            print(f"Move: {move}, Player: {player_id}")
            print(f"State After Move:\n{state_after}")
            print(f"Threats counted: {threats_created}")
        # --- End Debug --- 

        # A fork exists if the move leads to a state with 2 or more threats
        return threats_created >= 2 