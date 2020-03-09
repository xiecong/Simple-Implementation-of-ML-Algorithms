import numpy as np
from nn_layers import FullyConnect, Activation, Conv
# Double deep q learning (DQN) for Tic Tac Toe
n_size = 5
n_connect = 4


def is_done(board):
    for i in range(n_size * n_size):
        x, y = i % n_size, i // n_size
        x_end = x + n_connect
        x_rev_end = x - n_connect
        y_end = y + n_connect
        if (
                x_end <= n_size and abs(board[y, x:x_end].sum()) == n_connect
        ) or (
                y_end <= n_size and abs(board[y:y_end, x].sum()) == n_connect
        ) or (
                x_end <= n_size and y_end <= n_size and abs(
                    board[range(y, y_end), range(x, x_end)].sum()) == n_connect
        ) or (
                x_rev_end >= -1 and y_end <= n_size and abs(
                    board[range(y, y_end), range(x, x_rev_end, -1)].sum()) == n_connect
        ):
            return board[y, x]
    return 0


def transform_action(action):
    y = action // n_size
    x = action % n_size
    pos = [
        (y, x), (x, n_size - 1 - y), (n_size - 1 -
                                      y, n_size - 1 - x), (n_size - 1 - x, y),
        (y, n_size - 1 - x), (n_size - 1 - x,
                              n_size - 1 - y), (n_size - 1 - y, x), (x, y)
    ]
    return np.array([y * n_size + x for y, x in pos])


class NN(object):

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def gradient(self, grad_loss):
        grad = grad_loss
        for layer in self.layers[::-1]:
            grad = layer.gradient(grad)
        return grad

    def backward(self):
        for layer in self.layers:
            layer.backward()

    def copy_weights(self, nn):
        for layer1, layer2 in zip(self.layers, nn.layers):
            if isinstance(layer1, FullyConnect):
                layer1.w = layer2.w.copy()
                layer1.b = layer2.b.copy()


class DQN(object):

    def __init__(self, eps=1):
        self.n_episodes = 1000
        self.batch_size = 32
        self.n_epochs = 300
        self.gamma = 0.9
        self.eps = eps
        self.eps_decay = 0.999
        lr = 1e-5
        self.policy_net = NN([
            Conv((2, n_size, n_size), k_size=n_connect,
                 k_num=128, optimizer='RMSProp'),
            Activation(act_type='ReLU'),
            FullyConnect([128, n_size - n_connect + 1, n_size - n_connect + 1], [32],
                         lr=lr, optimizer='RMSProp'),
            Activation(act_type='ReLU'),
            FullyConnect([32], [16], lr=lr, optimizer='RMSProp'),
            Activation(act_type='ReLU'),
            FullyConnect([16], [n_size * n_size], lr=lr, optimizer='RMSProp'),
        ])
        self.target_net = NN([
            Conv((2, n_size, n_size), k_size=n_connect,
                 k_num=128, optimizer='RMSProp'),
            Activation(act_type='ReLU'),
            FullyConnect([128, n_size - n_connect + 1, n_size - n_connect + 1], [32],
                         lr=lr, optimizer='RMSProp'),
            Activation(act_type='ReLU'),
            FullyConnect([32], [16], lr=lr, optimizer='RMSProp'),
            Activation(act_type='ReLU'),
            FullyConnect([16], [n_size * n_size], lr=lr, optimizer='RMSProp'),
        ])

        self.states = np.zeros((0, 2, n_size, n_size))
        self.next_states = np.zeros((0, 2, n_size, n_size))
        self.actions = np.zeros(0).astype(int)
        self.rewards = np.zeros(0)
        self.unfinish_mask = np.zeros(0)

    def replay(self):
        permut = np.random.permutation(
            self.n_epochs * self.batch_size).reshape([self.n_epochs, self.batch_size])
        loss = 0
        for batch_idx in permut:
            action_pos = self.actions[batch_idx]

            this_q = np.zeros((self.batch_size, n_size * n_size))
            this_q[range(self.batch_size), action_pos] = self.policy_net.forward(
                self.states[batch_idx])[range(self.batch_size), action_pos]

            targets = np.zeros((self.batch_size, n_size * n_size))
            next_q = np.amax(self.target_net.forward(
                self.next_states[batch_idx]), axis=1)
            targets[range(self.batch_size), action_pos] = self.rewards[
                batch_idx] + self.unfinish_mask[batch_idx] * self.gamma * next_q

            grad = this_q - targets
            loss += np.square(grad).mean()
            self.policy_net.gradient(grad)
            self.policy_net.backward()
        print('loss', loss / self.n_epochs)

    def eps_greedy(self, state):
        valid_mask = 1 - state[0, 0, :, :].flatten() - \
            state[0, 1, :, :].flatten()
        preds = self.policy_net.forward(state)[0]
        max_idx = np.argmax(preds * valid_mask -
                            (1 - valid_mask) * np.finfo(float).max)
        m = sum(valid_mask)
        p = self.eps / m * valid_mask
        p[max_idx] = 1 - self.eps + self.eps / m
        return np.random.choice(n_size * n_size, 1, p=p)[0]

    def fit(self):
        training_size = self.n_epochs * self.batch_size
        for iteration in range(self.n_episodes):
            self.eps *= self.eps_decay
            boards = np.zeros((8, n_size * n_size))
            winner = 0
            n_moves = 0

            for move in range(n_size * n_size):
                n_moves += 1
                player = move % 2 * 2 - 1
                for board in boards:
                    self.states = np.append(self.states, np.array(
                        [[(board == player).reshape(n_size, n_size), (board == -player).reshape(n_size, n_size)]]), axis=0)[-training_size:]
                action_pos = self.eps_greedy(
                    np.array([[(boards[0] == player).reshape(n_size, n_size), (boards[0] == -player).reshape(n_size, n_size)]]))
                action_list = transform_action(action_pos)
                boards[range(8), action_list] = player
                for action, board in zip(action_list, boards):
                    self.actions = np.append(
                        self.actions, action)[-training_size:]
                    self.next_states = np.append(self.next_states, np.array(
                        [[(board == player).reshape(n_size, n_size), (board == -player).reshape(n_size, n_size)]]), axis=0)[-training_size:]
                winner = is_done(boards[0].reshape((n_size, n_size)))
                if abs(winner) == 1:
                    break

            this_mask, this_rewards = np.ones(n_moves), np.zeros(n_moves)
            this_mask[[-2, -1]] = np.array([0, 0])
            this_rewards[
                [-2, -1]] = np.array([-1 * abs(winner) + (1 - abs(winner)) * 0, 1 * abs(winner) + (1 - abs(winner)) * 0])
            self.unfinish_mask = np.append(
                self.unfinish_mask, np.repeat(this_mask, 8))[-training_size:]
            self.rewards = np.append(
                self.rewards, np.repeat(this_rewards, 8))[-training_size:]

            if self.states.shape[0] >= training_size:
                self.replay()
            if iteration % 50 == 0:
                self.target_net.copy_weights(self.policy_net)

            print('game', iteration, 'winner', winner,
                  'moves', move, 'eps', self.eps)


def test_against_random(dqn):
    game_records = [0, 0, 0]
    dqn.eps = 0

    for iteration in range(1000):
        board = np.zeros((n_size * n_size))
        records = np.zeros((n_size * n_size))
        winner = 0
        n_moves = 0

        for move in range(n_size * n_size):
            n_moves += 1
            player = move % 2 * 2 - 1
            if((int(iteration >= 500) + move) % 2 == 0):
                action_pos = dqn.eps_greedy(
                    np.array([[(board == player).reshape(n_size, n_size), (board == -player).reshape(n_size, n_size)]]))
            else:
                action_pos = np.random.choice(
                    n_size * n_size, 1, p=(1 - np.abs(board)) / (1 - abs(board)).sum())[0]
            board[action_pos] = player
            records[action_pos] = n_moves
            winner = is_done(board.reshape((n_size, n_size)))
            if abs(winner) == 1:
                break

        game_records[int(winner if iteration < 500 else -winner) + 1] += 1
        print('game', iteration, 'winner', winner, 'moves', move)
        print(records.reshape((n_size, n_size)))
    print('dqn win draw lose:', game_records)


def main():
    dqn = DQN()
    dqn.fit()
    test_against_random(dqn)

if __name__ == "__main__":
    main()
