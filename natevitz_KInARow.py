'''
natevitz_KInARow.py
Authors: Vitzthum, Nathan; Liu, Grace

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 415, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

from agent_base import KAgent
import game_types
from game_types import State, Game_Type

AUTHORS = 'Nathan Vitzthum and Grace Liu' 
 
import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.
import random

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Gerry'
        if twin: self.nickname += 'Flo'
        self.long_name = 'Gerald Flortinstein'
        if twin: self.long_name += ' Flora Flortinstein'
        self.persona = 'Provocative'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".
        self.alpha_beta_cutoffs_this_turn = 0
        self.num_static_evals_this_turn = 0
        self.zobrist_table_num_entries_this_turn = 0
        self.zobrist_table_num_hits_this_turn = 0
        self.current_game_type = None
        self.game_history = []  # List of State objects
        self.my_utterances = []  # List of strings
        self.opponent_utterances = []  # List of strings

    def introduce(self):
        intro = '\nMy name is Gerald Flortinstein.\n'+\
            '"Nathan and Grace" made me.\n'+\
            'I\'m ready to kick some tic-tac-toe butt!\n'
        if self.twin: intro += 'Hi I\'m Flo, I\'m the TWIN.\n'+\
                                ' I hope you have a great game!\n'
        return intro

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(self, game_type, what_side_to_play, opponent_nickname, 
                expected_time_per_move = 0.1, utterances_matter=True):      
        # If False, just return 'OK' for each utterance,
        # or something simple and quick to compute
        # and do not import any LLM or special APIs.
        # During the tournament, this will be False..

        if utterances_matter:
           pass
           # Optionally, import your LLM API here.
           # Then you can use it to help create utterances.

        self.current_game_type = game_type
        self.playing = what_side_to_play
        self.opponent = opponent_nickname
        self.time_limit = expected_time_per_move
        return "OK"     
   
    # The core of your agent's ability should be implemented here:             
    def make_move(self, current_state, current_remark, time_limit=1.0,
                  autograding=False, use_alpha_beta=True,
                  use_zobrist_hashing=False, max_ply=3,
                  special_static_eval_fn=None, order_moves=False):
        print("make_move has been called")

        self.alpha_beta_cutoffs_this_turn = 0
        self.num_static_evals_this_turn = 0

        start_time = time.time()
        best_move, _ = self.minimax(current_state, max_ply, pruning=use_alpha_beta, 
                                    alpha=float('-inf'), beta=float('inf'), 
                                    start_time=start_time, time_limit=time_limit,
                                    special_static_eval_fn=special_static_eval_fn, order_moves=order_moves)

        end_time = time.time()
        new_state = self.perform_move(current_state, best_move)
        # ADDED: Store the current state to game history
        self.game_history.append(current_state)
        self.game_history[-1].move_made = best_move

        # Store the stats from this move
        self.last_move_stats = {
            'time_taken': end_time - start_time,
            'static_evals': self.num_static_evals_this_turn,
            'alpha_beta_cutoffs': self.alpha_beta_cutoffs_this_turn,
            'zobrist_entries': self.zobrist_table_num_entries_this_turn,
            'zobrist_hits': self.zobrist_table_num_hits_this_turn
        }

        new_remark = self.select_utterance(current_state, best_move)

        # ADDED:  Check if the opponent asked for a game summary
        if current_remark == "What's your take on the game so far?":
            new_remark = self.generate_game_summary() # create the game summary

        # ADDED:  Check if the opponent asked for an explanation
        if current_remark == "Tell me how you did that":
            new_remark = self.explain_last_move() # create the explanation

        # need to use special evaluation function if autograding is true
        if autograding:
            stats = [self.alpha_beta_cutoffs_this_turn,
                     self.num_static_evals_this_turn,
                     self.zobrist_table_num_entries_this_turn,
                     self.zobrist_table_num_hits_this_turn]
            return [[best_move, new_state] + stats, new_remark]
        else:
            return [[best_move, new_state], new_remark]
        

    def select_utterance(self, current_state, best_move):
        """Selects an utterance based on the game state."""
        score = self.static_eval(current_state, self.current_game_type)

        if (self.twin):
            # Winning utterances
            if (score > (self.current_game_type.k * 20) and self.playing == 'X') or (score < (-20 * self.current_game_type.k) and self.playing == 'O'): #Tuned
                utterances = [
                    "I believe in you",
                    "Don't beat yourself up",
                    "There's always next game :)",
                ]
            #Losing utterances
            elif (score < (-20 * self.current_game_type.k) and self.playing == 'X') or (score > (20 * self.current_game_type.k) and self.playing == 'O'): 
                utterances = [
                    "Wow you're so good",
                    "I need to take notes on your strategy",
                    "Maybe I should just give up.",
                ]
            else:
                utterances = [
                    "Interesting...",
                    "I'm considering my options.",
                    "Here's my move.",
                ]
        else:
            # Winning utterances
            if (score > (self.current_game_type.k * 20) and self.playing == 'X') or (score < (-20 * self.current_game_type.k) and self.playing == 'O'): 
                utterances = [
                    "I'm crushing you!",
                    "This is too easy.",
                    "Do you even know how to play?",
                    "Here's some tissues",
                    "Sorry not all of us can be good at this game"
                ]
            #Losing utterances
            elif (score < (-20 * self.current_game_type.k) and self.playing == 'X') or (score > (20 * self.current_game_type.k) and self.playing == 'O'):
                utterances = [
                    "I'm just giving you a chance",
                    "Are you using an AI?",
                    "Must be beginner's luck",
                    "Just wait until I start trying"
                ]
            else:
                utterances = [
                    "This game is putting me to sleep",
                    "Do something cool for once",
                    "Give me a challenge",
                ]
        return random.choice(utterances)
        
    def explain_last_move(self):
        """Generates a detailed explanation of the last move."""
        if not self.last_move_stats:
            return "I'm sorry, I don't have information about the last move yet."

        stats = self.last_move_stats
        explanation = (
            "Okay, here's how I came up with that move:\n"
            f"I spent {stats['time_taken']:.4f} seconds thinking about it.\n"
            f"During that time, I evaluated {stats['static_evals']} board states using my static evaluation function.\n"
            f"I also performed {stats['alpha_beta_cutoffs']} alpha-beta cutoffs, which helped me prune the search space.\n"
        )

        if stats['zobrist_entries'] > 0:
            explanation += (
                f"My Zobrist hashing table had {stats['zobrist_entries']} entries, and I had {stats['zobrist_hits']} hits, which sped up the search.\n"
            )
        else:
            explanation += "I didn't use Zobrist hashing for this move.\n"

        explanation += "I used a minimax search with a depth of 3."

        return explanation


    def generate_game_summary(self):
        """Generates a story of the game so far, including a prediction."""
        story = "So far, the game has been quite interesting. From the beginning, "
        
        # Build the story from the game history
        for i, state in enumerate(self.game_history):
            move = state.move_made
            player = 'X' if i % 2 == 0 else 'O' 
            if move:
                story += f"Player {player} played at position {move}, "
            else:
                story += "The game began, "
        
        story += "leading to the current board state. "
        
        # Make a prediction about who will win (using static evaluation)
        current_state = self.game_history[-1] #The most recent state
        score = self.static_eval(current_state, self.current_game_type)
        
        if score > 100:  #Heuristic threshold for winning
            story += f"Based on my analysis, I predict that {self.playing} is likely to win."
        elif score < -100:
            opponent = 'O' if self.playing == 'X' else 'X'
            story += f"Based on my analysis, I predict that {opponent} is likely to win."
        else:
            story += "The game seems to be heading towards a draw."
        
        return story


    def minimax(self, state, depth_remaining, pruning=False,
            alpha=float('-inf'), beta=float('inf'),
            start_time=None, time_limit=None,
            order_moves=False, special_static_eval_fn=None):
        if time.time() - start_time > time_limit:
            return None, float('-inf')

        if depth_remaining == 0 or self.is_terminal(state):
            if special_static_eval_fn:
                score = special_static_eval_fn(state)
            else:
                self.num_static_evals_this_turn += 1
                score = self.static_eval(state, self.current_game_type)
            return None, score

        moves = self.get_legal_moves(state)
        if not moves:
            if special_static_eval_fn:
                score = special_static_eval_fn(state)
            else:
                self.num_static_evals_this_turn += 1
                score = self.static_eval(state, self.current_game_type)
            return None, score

        # Apply move ordering
        if order_moves:
            ordered_moves = self.order_moves(state, moves)
        else: 
            ordered_moves = moves

        if state.whose_move == 'X':
            best_score = float('-inf')
            best_move = None
            for move in ordered_moves:
                new_state = self.perform_move(state, move)
                _, score = self.minimax(new_state, depth_remaining - 1, pruning, alpha, beta, start_time, time_limit, special_static_eval_fn, order_moves)
                if score > best_score:
                    best_score = score
                    best_move = move
                if pruning:
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        self.alpha_beta_cutoffs_this_turn += 1
                        break
            return best_move, best_score
        else:
            best_score = float('inf')
            best_move = None
            for move in ordered_moves:
                new_state = self.perform_move(state, move)
                _, score = self.minimax(new_state, depth_remaining - 1, pruning, alpha, beta, start_time, time_limit, special_static_eval_fn, order_moves)
                if score < best_score:
                    best_score = score
                    best_move = move
                if pruning:
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        self.alpha_beta_cutoffs_this_turn += 1
                        break
        return best_move, best_score


    def static_eval(self, state, game_type):
        self.num_static_evals_this_turn += 1
        
        def evaluate_line(line):
            def score_player(player):
                unblocked = self.count_unblocked_sequence(player, line, game_type.k)
                return unblocked * (10 ** unblocked)
            return score_player('X') - score_player('O')

        return sum(evaluate_line(line) for line in self.extract_all_lines(state.board, game_type.k))

    def count_unblocked_sequence(self, player, sequence, k):
        max_unblocked = current_unblocked = open_spaces = 0
        
        for cell in sequence + [None]:
            if cell == player:
                current_unblocked += 1
                open_spaces += 1
            elif cell == " ":
                open_spaces += 1
            else:
                if open_spaces >= k:
                    max_unblocked = max(max_unblocked, current_unblocked)
                current_unblocked = open_spaces = 0
        
        return max_unblocked

    def extract_all_lines(self, board, k):
        rows, cols = len(board), len(board[0])
        
        def get_diagonals(reverse=False):
            diags = []
            for offset in range(-rows + 1, cols):
                diag = [board[i][i + offset] if reverse else board[i][cols - 1 - i - offset] 
                        for i in range(max(0, -offset), min(rows, cols - offset))]
                if len(diag) >= k:
                    diags.append(diag)
            return diags

        return (
            board +  # rows
            list(map(list, zip(*board))) +  # columns
            get_diagonals() +  # forward diagonals
            get_diagonals(reverse=True)  # backward diagonals
        )


    def get_legal_moves(self, state):
        moves = []
        board = state.board
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == ' ':
                    moves.append((i, j))
        return moves

    def perform_move(self, state, move):
        new_state = State(old=state)
        new_state.board[move[0]][move[1]] = state.whose_move
        new_state.whose_move = 'O' if state.whose_move == 'X' else 'X'
        return new_state

    def is_terminal(self, state):
        board = state.board
        size = len(board)

        # Check rows and columns
        for i in range(size):
            if self.check_line(board[i]):
                return True
            if self.check_line([board[j][i] for j in range(size)]): 
                return True

        # Check diagonals
        if self.check_line([board[i][i] for i in range(size)]): 
            return True
        if self.check_line([board[i][size - 1 - i] for i in range(size)]):
            return True

        # Check for draw (no empty spaces)
        for row in board:
            if ' ' in row:
                return False  

        return True  

    def check_line(self, line):
        return (line[0] != ' ' and len(set(line)) == 1)
    
    def order_moves(self, state, moves):
        move_scores = []
        for move in moves:
            new_state = self.perform_move(state, move)
            score = self.static_eval(new_state, self.current_game_type)
            move_scores.append((move, score))

        # Sort moves based on the score
        if self.playing == 'X':  # Maximizing player
            return [move for move, _ in sorted(move_scores, key=lambda x: x[1], reverse=True)]
        else:  # Minimizing player
            return [move for move, _ in sorted(move_scores, key=lambda x: x[1])]
    
    def test_move_ordering():
        agent = OurAgent()
        agent.prepare(game_types.TTT, 'X', 'TestOpponent')

        agent.prepare(game_types.FIAR, 'X', 'TestOpponent')

        # Test state 1: Early game with multiple good moves
        state1 = game_types.State(initial_state_data=[
            [['-',' ',' ',' ',' ',' ','-'],
            [' ','O',' ',' ',' ','X',' '],
            [' ',' ',' ',' ',' ',' ',' '],
            [' ',' ',' ','X',' ',' ',' '],
            [' ',' ',' ',' ',' ',' ',' '],
            [' ','X',' ',' ',' ','O',' '],
            ['-',' ',' ',' ',' ',' ','-']], "O"])

        # Test state 2: Mid game with a clear best move
        state2 = game_types.State(initial_state_data=[
            [['-','O','X',' ','X',' ','-'],
            [' ','O','O','X','O','X',' '],
            ['X','X','O','O','X',' ',' '],
            [' ','O','X','X','O','O',' '],
            [' ','X','O','O','X',' ',' '],
            [' ','O','X',' ','O','X',' '],
            ['-',' ',' ',' ',' ',' ','-']], "X"])

        for state in [state1, state2]:
            print(f"Testing state:\n{state}")
            results = agent.measure_minimax_performance(state, depth=4)
            
            print("Results without move ordering:")
            print(f"Time: {results['Without Ordering - Time']:.4f} seconds")
            print(f"Static Evaluations: {results['Without Ordering - Static Evals']}")
            print(f"Alpha-Beta Cutoffs: {results['Without Ordering - AB Cutoffs']}")
            
            print("\nResults with move ordering:")
            print(f"Time: {results['With Ordering - Time']:.4f} seconds")
            print(f"Static Evaluations: {results['With Ordering - Static Evals']}")
            print(f"Alpha-Beta Cutoffs: {results['With Ordering - AB Cutoffs']}")
            
            print("\n")


