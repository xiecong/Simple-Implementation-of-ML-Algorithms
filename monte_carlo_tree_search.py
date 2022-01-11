import numpy as np
from minimax import MiniMax, RandomMove  # for testing purpose
# implements Monte Carlo tree search for Tic Tac Toe / Gomoku


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
    for i in range(100):
        idx = [0, 1]  # np.random.permutation([0, 1]).astype(int)
        board, winner = play([agents[idx[0]], agents[idx[1]]])
        game_records[-int(winner) * (2 * idx[0] - 1) + 1] += 1
    return game_records


class MCTSNode(object):

    def __init__(self, board):
        self.board = board
        self.simulations = [0, 0, 0]  # lose/draw/win
        self.n_visit = 0
        self.children = {}
        self.score = 0
        self.done = np.abs(board).sum() == board.shape[
            0] or is_done(board.reshape(n_size, n_size)) != 0

    def update(self, result):
        self.simulations[result + 1] += 1  # -1/0/1 -> lose/draw/win (0,1,2)
        self.n_visit += 1
        self.score = (self.simulations[2] + 1 * self.simulations[1]) / self.n_visit  # 1 for draw


class MCTS(object):

    def __init__(self):
        self.cache = {}
        self.rm = RandomMove()
        self.n_iteration = 10 * n_size * n_size

    def legal_moves(self, board):
        return [i for i in range(n_size * n_size) if board[i] == 0]

    def selection(self, node):
        max_uct = -np.inf
        next_moves = []
        for move in self.legal_moves(node.board):
            score = node.children[move].score if move in node.children else 0
            child_visits = node.children[
                move].n_visit if move in node.children else 1e-4
            this_uct = score + np.sqrt(2 * np.log(node.n_visit) / child_visits)
            if max_uct < this_uct:
                next_moves = [move]
                max_uct = this_uct
            elif max_uct == this_uct:
                next_moves.append(move)
        return np.random.choice(next_moves)

    def simulation(self, board, player):  # todo add heuristics
        winner = is_done(board.reshape((n_size, n_size)))
        while np.abs(winner) == 0 and np.abs(board).sum() < board.shape[0]:
            board[self.rm.act(board, player)] = player
            winner = is_done(board.reshape((n_size, n_size)))
            player = -player
        return winner

    def search(self, root_node, player):
        for _ in range(self.n_iteration):
            parents = [root_node]
            node = root_node
            while not node.done and node.n_visit > 0:
                # selection
                next_move = self.selection(node)
                if next_move not in node.children:  # expansion
                    child_board = node.board.copy()
                    child_board[next_move] = -(node.board.sum() * 2 + 1)
                    # unable to share cache since #visits are not consistent
                    node.children[next_move] = MCTSNode(child_board)
                node = node.children[next_move]
                parents.append(node)
            # simulation
            result = self.simulation(node.board.copy(), node.board.sum() * 2 + 1)
            # backpropagation
            for p in parents[::-1]:
                this_player = p.board.sum() * 2 + 1
                p.update(result * this_player)

    def act(self, board, player):
        board_str = ''.join([str(int(i)) for i in board])
        if board_str not in self.cache:
            self.cache[board_str] = MCTSNode(board.copy())
        node = self.cache[board_str]
        self.search(node, player)
        v_max = np.amax([c.n_visit for m, c in node.children.items()])
        return np.random.choice([m for m, c in node.children.items() if c.n_visit == v_max])


def main():
    minimax = MiniMax(max_depth=9)
    mcts = MCTS()
    random = RandomMove()
    test([minimax, mcts])
    test([mcts, minimax])

    print('\t\t\t\twin/draw/lose')
    print('mcts vs. mcts', test([mcts, mcts]))
    print('random vs. mcts', test([random, mcts]))
    print('mcts vs. random', test([mcts, random]))
    print('minimax vs. mcts', test([minimax, mcts]))
    print('mcts vs. minimax', test([mcts, minimax]))

if __name__ == "__main__":
    main()
