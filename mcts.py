import numpy as np
# implements Minimax and Monte Carlo tree search for tic tac toe


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


def play(agents):
    board = np.zeros(n_size * n_size)
    record = np.zeros(n_size * n_size)
    winner = 0
    n_moves = 0

    for move in range(n_size * n_size):
        n_moves += 1
        player = move % 2 * 2 - 1
        action_pos = agents[move % 2].act(board, player)
        record[action_pos] = n_moves
        board[action_pos] = player
        winner = is_done(board.reshape((n_size, n_size)))
        if abs(winner) == 1:
            break
    return record.reshape((n_size, n_size)), winner


def test(agents):
    game_records = [0, 0, 0]
    for i in range(1000):
        idx = [0, 1]  # np.random.permutation([0, 1]).astype(int)
        board, winner = play([agents[idx[0]], agents[idx[1]]])
        game_records[-int(winner) * (2 * idx[0] - 1) + 1] += 1
    return game_records


class RandomMove(object):

    def act(self, board, player):
        return np.random.choice(n_size * n_size, 1, p=(1 - np.abs(board)) / (1 - abs(board)).sum())[0]


class MiniMax(object):

    def __init__(self):
        self.board_memory = {}

    def value(self, board, player):
        board_str = ''.join([str(int(i)) for i in board])
        if board_str in self.board_memory:  # calculated before
            return self.board_memory[board_str]
        winner = is_done(board.reshape(n_size, n_size))
        if np.abs(board).sum() == board.shape[0] or winner != 0:  # game end
            return -1, winner * player
        values = np.ones(board.shape[0]) * -2
        for i in range(board.shape[0]):
            if board[i] != 0:
                continue
            board[i] = player
            _, v = self.value(board, -player)
            values[i] = -v
            board[i] = 0
        vmax = np.amax(values)
        self.board_memory[board_str] = ([i for i in range(
            board.shape[0]) if values[i] == vmax], vmax)
        return self.board_memory[board_str]

    def act(self, board, player):
        return np.random.choice(self.value(board, player)[0])


class MCTS(object):
    def __init__(self):
    	pass

    def value(self, board, player):
    	pass

    def act(self, board, player):
        pass

def main():
    minimax1 = MiniMax()
    minimax2 = MiniMax()
    random = RandomMove()
    print('\t\t\t\twin/draw/lose')
    print('minimax vs. random', test([minimax1, random]))
    print('random vs. minimax', test([random, minimax1]))
    print('minimax vs. minimax', test([minimax1, minimax2]))


if __name__ == "__main__":
    main()