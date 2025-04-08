# Training System Documentation

The training system provides the framework for the AI agent to learn and improve its Tic Tac Toe playing strategy autonomously through self-play. By repeatedly playing games against itself or other versions of itself, the agent gathers experience, processes outcomes, and updates its internal knowledge base (memory and evaluations).

The primary component of this system is typically a trainer class, such as `SelfPlayTrainer` located in `training/self_play.py`.

---

## Self-Play Training (`SelfPlayTrainer`)

The `SelfPlayTrainer` orchestrates the self-play learning process.

**Key Functions and Responsibilities:**

1.  **Agent Initialization**: Creates and manages the agent(s) involved in the training process. Often, this involves training a single agent that plays against itself, acting as both Player 1 (X) and Player 2 (O) in alternating games or even making moves for both sides within the same game.

2.  **Game Simulation**: Manages the execution of Tic Tac Toe games. This involves:
    *   Initializing the `GameEnvironment` for each new game.
    *   Requesting moves from the current player agent based on the game state.
    *   Applying the selected moves to the environment.
    *   Detecting game termination (win, loss, or draw).

3.  **Experience Collection**: Gathers the data generated during each game. While the `Agent` itself often uses its `ShortTermMemory` to record the sequence of states and moves during the game, the trainer ensures this process occurs correctly within the game loop.

4.  **Agent Updates**: After each game concludes, the trainer signals the agent(s) to process the results. This typically involves calling the agent's `update_from_game_result` method, which in turn utilizes the `ExperienceProcessor` to analyze the game stored in `ShortTermMemory` and update the `ExperienceMemory` and `StrategicMemory`.

5.  **Training Loop Management**: Controls the overall training duration, usually defined by a specific number of games or epochs. It manages the iterative process of playing games and updating the agent.

6.  **Performance Tracking (Optional)**: Monitors the agent's progress during training. This can include metrics like:
    *   Win/loss/draw rates over sliding windows.
    *   Changes in pattern values or state evaluations.
    *   Game length statistics.

Through this automated cycle of play, reflection (experience processing), and knowledge update, the agent progressively learns optimal strategies, pattern recognition, and positional awareness for Tic Tac Toe. 