import numpy as np
from nn_layers import FullyConnect, Activation, Conv
from minimax import MiniMax, RandomMove
# Double deep q learning (DQN) for Tic Tac Toe / Gomoku


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
            if isinstance(layer1, FullyConnect) or isinstance(layer1, Conv):
                layer1.w = layer2.w.copy()
                layer1.b = layer2.b.copy()


class DQN(object):

    def __init__(self, eps=1):
        self.n_episodes = 300
        self.batch_size = 32
        self.n_epochs = 200
        self.training_size = self.n_epochs * self.batch_size
        self.gamma = 0.99
        self.eps = eps
        self.eps_decay = 0.99
        lr = 0.002
        self.policy_net, self.target_net = [NN([
            Conv((3, n_size, n_size), k_size=n_connect,
                 k_num=16, optimizer='RMSProp'),
            Activation(act_type='LeakyReLU'),
            FullyConnect([16, n_size - n_connect + 1, n_size - n_connect + 1], [16],
                         lr=lr, optimizer='RMSProp'),
            Activation(act_type='LeakyReLU'),
            FullyConnect([16], [16], lr=lr, optimizer='RMSProp'),
            Activation(act_type='LeakyReLU'),
            FullyConnect([16], [16], lr=lr, optimizer='RMSProp'),
            Activation(act_type='LeakyReLU'),
            FullyConnect([16], [16], lr=lr, optimizer='RMSProp'),
            Activation(act_type='LeakyReLU'),
            FullyConnect([16], [n_size * n_size], lr=lr, optimizer='RMSProp'),
            # Activation(act_type='Tanh'),
        ]) for _ in range(2)]
        self.states = np.zeros((0, 3, n_size, n_size))
        self.next_states = np.zeros((0, 3, n_size, n_size))
        self.actions = np.zeros(0).astype(int)
        self.rewards = np.zeros(0)
        self.unfinish_mask = np.zeros(0)
        self.weights = np.zeros(0)

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

            grad = (this_q - targets) * self.weights[batch_idx].reshape(-1, 1)
            loss += np.square(grad).mean()
            self.policy_net.gradient(grad)
            self.policy_net.backward()
        print('loss', loss / self.n_epochs)

    def act(self, board, player):
        state = np.array([[(board == player).reshape(
            n_size, n_size), (board == -player).reshape(n_size, n_size), (board == 0).reshape(n_size, n_size)]])
        return self.eps_greedy(state)

    def eps_greedy(self, state):
        valid_mask = 1 - state[0, 0, :, :].flatten() - \
            state[0, 1, :, :].flatten()
        preds = self.policy_net.forward(state)[0]
        max_idx = np.argmax(preds * valid_mask -
                            (1 - valid_mask) * np.finfo(float).max)
        m = sum(valid_mask)
        p = self.eps / m * valid_mask
        p[max_idx] = 1 - self.eps + self.eps / m
        return np.random.choice(n_size * n_size, p=p)

    def fit(self):
        random = RandomMove()
        minimax = MiniMax(max_depth=9)
        agents = [minimax, self]
        while self.states.shape[0] < self.training_size:
            # np.random.shuffle(agents)
            play(agents, self)
        for iteration in range(self.n_episodes):
            self.eps *= self.eps_decay
            # np.random.shuffle(agents)
            play(agents, self)
            print('iteration:', iteration, 'eps:', self.eps)
            for i in range(10):
                self.replay()
            if iteration % 10 == 0:
                self.target_net.copy_weights(self.policy_net)
            temp_eps = self.eps
            self.eps = 0
            print('\t\t\t\twin/draw/lose')
            print('minimax vs. dqn', test([minimax, self]))
            print('dqn vs. minimax', test([self, minimax]))
            print('random vs. dqn', test([random, self]))
            print('dqn vs. random', test([self, random]))
            self.eps = temp_eps

    def save_play(self, saved_actions, saved_states, winner, n_moves, saved_weights):
        self.actions = np.append(self.actions, np.array(
            saved_actions))[-self.training_size:]
        self.states = np.append(self.states, np.array(
            saved_states), axis=0)[-self.training_size:]
        self.next_states = np.append(
            self.next_states, np.array(saved_states[16:]), axis=0)
        self.next_states = np.append(self.next_states, np.zeros(
            (16, 3, n_size, n_size)), axis=0)[-self.training_size:]
        this_mask, this_rewards = np.ones(n_moves), np.zeros(n_moves)
        this_mask[[-2, -1]] = np.array([0, 0])
        this_rewards[[-2, -1]] = np.array([-1 * abs(winner) + (
            1 - abs(winner)) * 1, 1 * abs(winner) + (1 - abs(winner)) * 1])
        self.unfinish_mask = np.append(
            self.unfinish_mask, np.repeat(this_mask, 8))[-self.training_size:]
        self.rewards = np.append(self.rewards, np.repeat(
            this_rewards, 8))[-self.training_size:]
        self.weights = np.append(self.weights, np.array(
            saved_weights))[-self.training_size:]


def play(agents, cache=None):
    boards = np.zeros((8, n_size * n_size)).astype(int)
    record = np.zeros(n_size * n_size)
    winner = 0
    n_moves = 0
    saved_actions = []
    saved_states = []
    saved_weights = []
    for move in range(n_size * n_size):
        n_moves += 1
        player = move % 2 * 2 - 1
        action_pos = agents[move % 2].act(boards[0], player)
        record[action_pos] = n_moves
        action_list = transform_action(action_pos)
        for action, current_board in zip(action_list, boards):
            saved_actions.append(action)
            saved_states.append([
                (current_board == player).reshape(n_size, n_size),
                (current_board == -player).reshape(n_size, n_size),
                (current_board == 0).reshape(n_size, n_size)
            ])
            # only do q learning update for the dqn's move
            saved_weights.append(1 if isinstance(agents[move % 2], DQN) else 0)
        boards[range(8), action_list] = player
        winner = is_done(boards[0].reshape((n_size, n_size)))
        if abs(winner) == 1:
            break
    if cache is not None:
        cache.save_play(saved_actions, saved_states,
                        winner, n_moves, saved_weights)
    return record.reshape((n_size, n_size)), winner


def test(agents):
    game_records = [0, 0, 0]
    for i in range(100):
        idx = [0, 1]  # np.random.permutation([0, 1]).astype(int)
        board, winner = play([agents[idx[0]], agents[idx[1]]])
        game_records[-int(winner) * (2 * idx[0] - 1) + 1] += 1
    return game_records


def main():
    dqn = DQN()
    dqn.fit()

if __name__ == "__main__":
    main()
