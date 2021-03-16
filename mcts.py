import numpy as np
# implements Minimax and Monte Carlo tree search for Tic Tac Toe / Gomoku


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
    board = np.zeros(n_size * n_size).astype(int)
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

    def __init__(self, max_depth=4):
        self.cache = {}
        self.max_depth = max_depth

    def heuristic(self, board, player):
        # a. in [1, -1], b. score(player1) = -score(player2)
        evals = [0, 0]  # for player -1 1
        for i in range(n_size * n_size):
            if board[i] == 0:
                continue
            evals[(board[i] + 1 ) // 2] += \
                (i % n_size < n_size - 1 and board[i] == board[i + 1]) + \
                (i + n_size < board.shape[0] - 1 and board[i] == board[i + n_size]) + \
                (i + n_size < board.shape[0] - 1 and i % n_size > 0 and board[i] == board[i + n_size - 1]) + \
                (i + n_size < board.shape[0] - 1 and i % n_size <
                 n_size - 1 and board[i] == board[i + n_size + 1])
        return (-evals[0] * player + evals[1] * player) / (evals[0] + evals[1] + 1)

    def score(self, board, player, depth):
        board_str = ''.join([str(int(i)) for i in board])
        if board_str in self.cache:  # cached before
            return self.cache[board_str]
        winner = is_done(board.reshape(n_size, n_size))
        if np.abs(board).sum() == board.shape[0] or winner != 0:  # game end
            return -1, winner * player
        # a value less than -1 so next step can pick a legal move
        board_scores = np.ones(board.shape[0]) * -2
        for i in range(board.shape[0]):
            if board[i] != 0:
                continue
            board[i] = player
            board_scores[i] = -self.score(board, -player, depth + 1)[
                1] if depth < self.max_depth else self.heuristic(board, player)
            board[i] = 0
        best_score = np.amax(board_scores)
        best_moves = [i for i in range(board.shape[0]) if board_scores[
            i] == best_score]
        if np.abs(best_score) == 1:  # certain win or lose rather than heuristics
            self.cache[board_str] = best_moves, best_score
        return best_moves, best_score

    def act(self, board, player):
        return np.random.choice(self.score(board, player, 0)[0])


class MCTS(object):

    def __init__(self, max_depth=4):
        self.cache = {}
        self.max_depth = max_depth

    def heuristic(self, board, player):  # todo
        pass

    def score(self, board, player, depth):  # todo
        pass

    def act(self, board, player):
        return np.random.choice(self.score(board, player)[0])


def main():
    minimax = MiniMax()
    mcts = MCTS()
    random = RandomMove()
    print('\t\t\t\twin/draw/lose')
    print('random vs. minimax', test([random, minimax]))
    print('minimax vs. minimax', test([minimax, minimax]))
    print('minimax vs. random', test([minimax, random]))


if __name__ == "__main__":
    main()
