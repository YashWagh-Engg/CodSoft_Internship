# ---------------- TIC TAC TOE AI ----------------

# Initialize board
board = [" " for _ in range(9)]

# Display board
def print_board():
    print()
    for row in [board[i*3:(i+1)*3] for i in range(3)]:
        print("| " + " | ".join(row) + " |")
    print()

# Check winner
def check_winner(player):
    win_conditions = [
        [0,1,2], [3,4,5], [6,7,8],     # rows
        [0,3,6], [1,4,7], [2,5,8],     # columns
        [0,4,8], [2,4,6]               # diagonals
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

# Check draw
def is_draw():
    return " " not in board

# Minimax algorithm
def minimax(is_maximizing):
    if check_winner("O"):
        return 1
    if check_winner("X"):
        return -1
    if is_draw():
        return 0

    if is_maximizing:
        best_score = -1000
        for i in range(9):
            if board[i] == " ":
                board[i] = "O"
                score = minimax(False)
                board[i] = " "
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = 1000
        for i in range(9):
            if board[i] == " ":
                board[i] = "X"
                score = minimax(True)
                board[i] = " "
                best_score = min(score, best_score)
        return best_score

# AI move
def ai_move():
    best_score = -1000
    best_move = 0
    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(False)
            board[i] = " "
            if score > best_score:
                best_score = score
                best_move = i
    board[best_move] = "O"

# Human move
def human_move():
    while True:
        move = input("Enter your move (1-9): ")
        if move.isdigit():
            move = int(move) - 1
            if 0 <= move < 9 and board[move] == " ":
                board[move] = "X"
                break
        print("Invalid move. Try again.")

# Main game loop
def play_game():
    print("Welcome to Tic-Tac-Toe!")
    print("You are X, AI is O")
    print("Positions are 1 to 9")

    print_board()

    while True:
        human_move()
        print_board()

        if check_winner("X"):
            print("🎉 You win!")
            break
        if is_draw():
            print("🤝 It's a draw!")
            break

        ai_move()
        print("AI move:")
        print_board()

        if check_winner("O"):
            print("🤖 AI wins! (Unbeatable)")
            break
        if is_draw():
            print("🤝 It's a draw!")
            break

play_game()