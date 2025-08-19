''' RandomPlayer.py
A player for the game of K-in-a-Row (on N by M board with forbidden squares.)

'''

from agent_base import KAgent
import game_types
from random import randint
GAME_TYPE = None

class OurAgent(KAgent):

    def __init__(self, twin=False):
        self.twin = twin
        self.nickname = 'Randy'
        if twin: self.nickname = "Randy-Junior"
        self.long_name = 'Random Walker'
        if twin: self.long_name = "Random Walker Junior"
        self.my_past_utterances = None
        self.opponent_past_utterances = None
        self.repeat_count = None
        self.utt_count = None
        
    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
            self,
            game_type,
            what_side_to_play,
            opponent_nickname,
            expected_time_per_move = 0.1, # Time limits can be
                                          # changed mid-game by the game master.
            utterances_matter = True):    # If False, just return 'OK' for each utterance.


       # Write code to save the relevant information in variables
       # local to this instance of the agent.
       # Game-type info can be in global variables.
       self.who_i_play = what_side_to_play
       self.opponent_nickname = opponent_nickname
       self.time_limit = expected_time_per_move
       global GAME_TYPE
       GAME_TYPE = game_type
       print("Oh, I love playing randomly at ", game_type.long_name)
       self.my_past_utterances = []
       self.opponent_past_utterances = []
       self.repeat_count = 0
       self.utt_count = 0
       if self.twin: self.utt_count = 5 # Offset the twin's utterances.

       return "OK"

    def introduce(self):
        if self.twin:
            remark = "Call me the Junior Random Walker."
        else:
            remark = "My name is "+self.long_name+". Or is it Walky Rander?"
        return remark

    def nickname(self): return self.nickname

    def make_move(self, state, last_utterance, time_limit):
        possibleMoves = successors_and_moves(state)
        myMove = chooseMove(possibleMoves)
        myUtterance = self.nextUtterance()
        newState, newMove = myMove
        return [[newMove, newState], myUtterance]

    def nextUtterance(self):
        if self.repeat_count > 1: return "I am randomed out now."
        n = len(UTTERANCE_BANK)
        if self.utt_count == n:
            self.utt_count = 0
            self.repeat_count += 1
        this_utterance = UTTERANCE_BANK[self.utt_count]
        self.utt_count += 1
        return this_utterance
   
# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances


# Figure out who the other player is.
# For example, other("X") = "O".
def other(p):
    if p=='X': return 'O'
    return 'X'

# Randomly choose a move.
def chooseMove(statesAndMoves):
    states, moves = statesAndMoves
    if states==[]: return None
    random_index = randint(0, len(states)-1)
    my_choice = [states[random_index], moves[random_index]]
    return my_choice

# The following is a Python "generator" function that creates an
# iterator to provide one move and new state at a time.
# It could be used in a smarter agent to only generate SOME of
# of the possible moves, especially if an alpha cutoff or beta
# cutoff determines that no more moves from this state are needed.
def move_gen(state):
    b = state.board
    p = state.whose_move
    o = other(p)
    mCols = len(b[0])
    nRows = len(b)

    for i in range(nRows):
        for j in range(mCols):
            if b[i][j] != ' ': continue
            news = do_move(state, i, j, o)
            yield [(i, j), news]

# This uses the generator to get all the successors.
def successors_and_moves(state):
    moves = []
    new_states = []
    for item in move_gen(state):
        moves.append(item[0])
        new_states.append(item[1])
    return [new_states, moves]

# Performa a move to get a new state.
def do_move(state, i, j, o):
            news = game_types.State(old=state)
            news.board[i][j] = state.whose_move
            news.whose_move = o
            return news
    
UTTERANCE_BANK = ["k",
                  "l"
                  ]


def test():
    global GAME_TYPE
    GAME_TYPE = game_types.TTT
    print(GAME_TYPE)
    h = OurAgent()
    print("I am ", h.nickname)
    
    ttt = GAME_TYPE.initial_state
    print("ttt initial state: ")
    print(ttt)
    print("successors_and_moves: ")
    print(successors_and_moves(ttt))

if __name__=="__main__":
    test()
