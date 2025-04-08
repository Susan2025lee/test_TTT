document.addEventListener('DOMContentLoaded', () => {
    const boardElement = document.getElementById('board');
    const messageElement = document.getElementById('message');
    const resetButton = document.getElementById('reset-button');
    let humanPlayerId = 1; // Default, will be updated from server
    let gameActive = true; // To disable clicks when game is over or agent thinking

    // --- Function to update the board display --- 
    function updateBoard(state) {
        boardElement.innerHTML = ''; // Clear previous board
        gameActive = !state.isGameOver; // Update game status

        // Add/remove class to disable clicks/hover when game over
        if (!gameActive) {
            boardElement.classList.add('game-over');
        } else {
            boardElement.classList.remove('game-over');
        }

        state.board.flat().forEach((cellValue, index) => {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            const row = Math.floor(index / 3);
            const col = index % 3;
            cell.dataset.row = row;
            cell.dataset.col = col;

            if (cellValue === 1) {
                cell.textContent = 'X';
                cell.classList.add('X');
            } else if (cellValue === -1) {
                cell.textContent = 'O';
                cell.classList.add('O');
            } else if (gameActive && state.currentPlayer === humanPlayerId) {
                 // Only add click listener to empty cells if it's human's turn and game active
                 cell.addEventListener('click', handleCellClick);
            }
            boardElement.appendChild(cell);
        });

        // Update message
        updateMessage(state);
    }

    // --- Function to update the message display --- 
    function updateMessage(state) {
        if (state.isGameOver) {
            if (state.winner === humanPlayerId) {
                messageElement.textContent = 'Congratulations! You Win!';
            } else if (state.winner === -humanPlayerId) {
                messageElement.textContent = 'Agent Wins!';
            } else {
                messageElement.textContent = 'It\'s a Draw!';
            }
        } else {
            if (state.currentPlayer === humanPlayerId) {
                messageElement.textContent = 'Your turn (X)';
            } else {
                messageElement.textContent = 'Agent\'s turn (O)... Thinking...';
            }
        }
    }

    // --- Function to handle player clicking a cell ---
    function handleCellClick(event) {
        if (!gameActive) return; // Ignore clicks if game is over or agent moving

        const cell = event.target;
        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);

        // Prevent clicking non-empty cells (extra check)
        if (cell.textContent !== '') return;
        
        console.log(`Human clicked: [${row}, ${col}]`);
        // Temporarily disable clicks while processing move
        gameActive = false; 
        messageElement.textContent = 'Processing your move...';
        
        // Send the move to the backend
        fetch('/move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ move: [row, col] })
        })
        .then(response => response.json())
        .then(newState => {
            console.log("Received state after move:", newState);
            if (newState.error) {
                 console.error("Server Error:", newState.error);
                 messageElement.textContent = `Error: ${newState.error}. Please reset.`;
                 // Don't re-enable game on server error
            } else {
                humanPlayerId = newState.humanPlayerId; // Ensure player ID is up to date
                updateBoard(newState); // Update board with the new state from server
                // Game active state will be set within updateBoard based on newState.isGameOver
            }
        })
        .catch(error => {
            console.error('Error sending move:', error);
            messageElement.textContent = 'Error communicating with server. Please reset.';
             // Keep game inactive on communication error
        });
    }

    // --- Function to handle reset button click --- 
    function handleResetClick() {
        console.log("Resetting game...")
        fetch('/reset', { method: 'POST' })
            .then(response => response.json())
            .then(initialState => {
                console.log("Received state after reset:", initialState);
                 if (initialState.error) {
                     console.error("Reset Error:", initialState.error);
                     messageElement.textContent = `Error: ${initialState.error}.`;
                 } else {
                    humanPlayerId = initialState.humanPlayerId;
                    updateBoard(initialState);
                 }
            })
            .catch(error => {
                console.error('Error resetting game:', error);
                messageElement.textContent = 'Error communicating with server.';
            });
    }

    // --- Initial Game Load --- 
    function loadInitialState() {
        console.log("Requesting initial state...")
        fetch('/get_state')
            .then(response => response.json())
            .then(initialState => {
                console.log("Received initial state:", initialState);
                 if (initialState.error) {
                     console.error("Initial State Error:", initialState.error);
                     messageElement.textContent = `Error: ${initialState.error}.`;
                 } else {
                    humanPlayerId = initialState.humanPlayerId;
                    updateBoard(initialState);
                 }
            })
            .catch(error => {
                console.error('Error fetching initial state:', error);
                messageElement.textContent = 'Error connecting to the game server.';
            });
    }

    // Add event listener for reset button
    resetButton.addEventListener('click', handleResetClick);

    // Load the initial game state when the page loads
    loadInitialState();
}); 