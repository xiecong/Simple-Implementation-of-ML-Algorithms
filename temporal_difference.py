import numpy as np
from nn_layers import FullyConnect, Activation, Conv
from minimax import MiniMax, RandomMove
# Temporal difference Q learning for Tic Tac Toe / Gomoku


n_size = 3
n_connect = 3


def is_done(board):
    for i in range(n_size * n_size):
        x, y = i % n_size, i // n_size
        x_end = x + n_connect
        x_rev_end = x - n_connect
        y_end = y + n_connect
        if (  # -
            x_end <= n_size and abs(board[y, x:x_end].sum()) == n_connect
        ) or (  # |
            y_end <= n_size and abs(board[y:y_end, x].sum()) == n_connect
        ) or (  # \
            x_end <= n_size and y_end <= n_size and abs(
                board[range(y, y_end), range(x, x_end)].sum()) == n_connect
        ) or (  # /
            x_rev_end >= -1 and y_end <= n_size and abs(
                board[range(y, y_end), range(x, x_rev_end, -1)].sum()) == n_connect
        ):
            return board[y, x]
    return 0


def transform_action(action):  # generating more board by flipping and rotating
    y = action // n_size
    x = action % n_size
    pos = [
        (y, x), (x, n_size - 1 - y), (n_size - 1 -
                                      y, n_size - 1 - x), (n_size - 1 - x, y),
        (y, n_size - 1 - x), (n_size - 1 - x,
                              n_size - 1 - y), (n_size - 1 - y, x), (x, y)
    ]
    return np.array([y * n_size + x for y, x in pos])


class TD(object):

    def __init__(self):
        self.q = {}
        self.draw_reward = 0.6
        self.alpha = 0.9
        self.gamma = 0.95

    def hash(self, board):
        hash_str = ''.join([str(i) for i in board.tolist()])
        if hash_str not in self.q:
            self.q[hash_str] = self.draw_reward * (1 - abs(board)) - abs(board)
        return hash_str

    def act(self, board, player=None):
    	if board[np.argmax(self.q[self.hash(board)])] != 0:
    		print('error')
    	return np.argmax(self.q[self.hash(board)])

    def fit(self):
        random = RandomMove()
        minimax = MiniMax(max_depth=9)
        agents = np.array([random, self])
        state = np.zeros(n_size * n_size)
        for i in range(20001):
            np.random.shuffle(agents)
            extended_boards, extended_actions, rewards, unfinished_flags, _ = play(
                agents)
            for board_sequence, action_sequence in zip(extended_boards, extended_actions):
                for state, next_state, action, reward, unfinished in zip(
                        board_sequence[
                            :-1], board_sequence[1:], action_sequence, rewards, unfinished_flags
                ):
                    state_hash = self.hash(state)
                    next_hash = self.hash(next_state)
                    self.q[state_hash][action] += self.alpha * (
                        reward + self.gamma * unfinished *
                        np.amax(self.q[next_hash]) - self.q[state_hash][action]
                    )
            if i % 1000 == 0:
                print(f'iteration {i}\t\t\twin/draw/lose')
                print('minimax vs. q learning', test([minimax, self]))
                print('q learning vs. minimax', test([self, minimax]))
                print('random vs. q learning', test([random, self]))
                print('q learning vs. random', test([self, random]))


def play(agents):
    boards = np.zeros((8, n_size * n_size)).astype(int)
    winner = 0
    saved_actions = []
    saved_states = []
    for move in range(n_size * n_size):
        player = move % 2 * 2 - 1
        action_pos = agents[move % 2].act(boards[0], player)
        action_list = transform_action(action_pos)
        if isinstance(agents[move % 2], TD):
            saved_actions.append(action_list)
            saved_states.append(boards.copy())
        boards[range(8), action_list] = player
        winner = is_done(boards[0].reshape((n_size, n_size)))
        if abs(winner) == 1:
            break
    saved_states.append(np.zeros((8, n_size * n_size)).astype(int))
    rewards = np.zeros(len(saved_actions))
    unfinished_flags = np.ones(len(saved_actions))
    rewards[-1] = winner * (2 * isinstance(agents[1], TD) - 1)
    unfinished_flags[-1] = 0
    return np.transpose(saved_states, (1, 0, 2)), np.transpose(saved_actions), rewards, unfinished_flags, winner


def test(agents):
    game_records = [0, 0, 0]
    for i in range(100):
        _, _, _, _, winner = play(agents)
        game_records[int(winner) + 1] += 1
    return game_records


def main():
    td = TD()
    td.fit()

if __name__ == "__main__":
    main()
