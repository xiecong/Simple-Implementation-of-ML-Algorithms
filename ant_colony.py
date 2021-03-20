import numpy as np
import matplotlib.pyplot as plt
# ant colony for traveling salesman problem


class ACO(object):

    def __init__(self, nodes):
        self.n_node = nodes.shape[0]
        self.pheromone = np.ones((self.n_node, self.n_node))
        self.n_ants = 50
        self.rho = 0.1
        self.alpha = 1
        self.beta = 2
        self.q = 50
        self.distance = np.array([
            np.sqrt(np.square(nodes[i] - nodes[j]).sum())
            for j in range(self.n_node) for i in range(self.n_node)
        ]).reshape(self.n_node, -1)
        self.dist_inv_beta = np.array([
            np.power(np.square(nodes[i] - nodes[j]).sum(), -self.beta / 2)
            if i != j else 0
            for j in range(self.n_node) for i in range(self.n_node)
        ]).reshape(self.n_node, -1)

    def generate_path(self):
        # pick random init node
        this_id, next_id = -1, np.random.choice(np.arange(self.n_node))
        visited_ids = [next_id]
        this_distance = 0
        for _ in range(self.n_node - 1):
            this_id = next_id
            # available_nodes
            p = np.power(self.pheromone[this_id],
                         self.alpha) * self.dist_inv_beta[this_id]
            p[visited_ids] = 0
            next_id = np.random.choice(self.n_node, 1, p=p / np.sum(p))[0]
            this_distance += self.distance[this_id, next_id]
            visited_ids.append(next_id)
        this_distance += self.distance[next_id, visited_ids[0]]
        visited_ids.append(visited_ids[0])
        return visited_ids, this_distance

    def update_pheromone(self, paths, dists):
        self.pheromone *= (1 - self.rho)
        for path, dist in zip(paths, dists):
            for this_id, next_id in zip(path[1:], path[:-1]):
                self.pheromone[
                    this_id, next_id] += self.q * dist

    def optimize(self):
        best_path = []
        min_distance = np.inf
        for _ in range(100):
            paths, dists = [], []
            for _ in range(self.n_ants):
                this_path, this_distance = self.generate_path()
                paths.append(this_path)
                dists.append(this_distance)
                if min_distance > this_distance:
                    best_path = this_path
                    min_distance = this_distance
            self.update_pheromone(paths, dists)
        return best_path, min_distance


def main():
    nodes = np.array([[565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0], [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0], [1605.0, 620.0], [1220.0, 580.0], [1465.0, 200.0], [1530.0, 5.0], [845.0, 680.0], [725.0, 370.0], [145.0, 665.0], [415.0, 635.0], [510.0, 875.0], [560.0, 365.0], [300.0, 465.0], [520.0, 585.0], [480.0, 415.0], [835.0, 625.0], [975.0, 580.0], [1215.0, 245.0], [
                     1320.0, 315.0], [1250.0, 400.0], [660.0, 180.0], [410.0, 250.0], [420.0, 555.0], [575.0, 665.0], [1150.0, 1160.0], [700.0, 580.0], [685.0, 595.0], [685.0, 610.0], [770.0, 610.0], [795.0, 645.0], [720.0, 635.0], [760.0, 650.0], [475.0, 960.0], [95.0, 260.0], [875.0, 920.0], [700.0, 500.0], [555.0, 815.0], [830.0, 485.0], [1170.0, 65.0], [830.0, 610.0], [605.0, 625.0], [595.0, 360.0], [1340.0, 725.0], [1740.0, 245.0]])
    aco = ACO(nodes)
    best_path, min_distance = aco.optimize()
    print('best path:', best_path, 'length:', min_distance)
    plt.scatter(nodes[:, 0], nodes[:, 1])

    pos = nodes[best_path]
    angles = pos[1:] - pos[:-1]
    plt.quiver(pos[:-1, 0], pos[:-1, 1], angles[:, 0], angles[:, 1],
               scale_units='xy', angles='xy', scale=1, width=0.004)
    plt.show()


if __name__ == "__main__":
    main()
