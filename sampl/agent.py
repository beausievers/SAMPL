import copy
import numpy as np


def interp_matrices(x, y, interp):
    return x - (x - y) * interp


class Agent(object):
    def __init__(self, id_string, op, learning_rate=0.2, print_log=True):
        if print_log:
            print(f'"Hello world, I\'m {id_string}."')
        self.id_string = id_string
        self.op = op
        self.learning_rate = learning_rate
        self.episode_op = None
        self.print_log = print_log

    def start_episode(self):
        if self.print_log:
            print(f"{self.id_string} is listening")
        self.episode_op = copy.deepcopy(self.op)

    def end_episode(self):
        if self.print_log:
            print(f"{self.id_string} stopped listening")
        self.op.graph.adj = interp_matrices(
            self.op.graph.adj, self.episode_op.graph.adj, self.learning_rate
        )
        self.episode_op = None

    def send(self, receivers, words):
        if self.print_log:
            print(f"{self.id_string} says \"{' '.join(words)}\"")
        for receiver in receivers:
            receiver.receive(words)

    def receive(self, words):
        if self.print_log:
            print(f"{self.id_string} received {words}")
        if self.episode_op is None:
            if self.print_log:
                print(f"{self.id_string} wasn't listening")
            return
        if self.print_log:
            for word in words:
                if word not in self.op.graph.labels:
                    print(f'{self.id_string} doesn\'t know the word "{word}"')
        words = [word for word in words if word in self.op.graph.labels]

        activations = np.zeros(len(self.op.graph.labels))
        ix = [self.op.graph.labels.index(word) for word in words]
        activations[ix] = 1
        self.episode_op.activate_replace(activations)
        if self.print_log:
            print(f"{self.id_string} updated their episode graph weights")

    def spontaneous_word_pair(self):
        assert self.episode_op is not None
        mean_in = np.mean(self.episode_op.graph.adj, axis=0)
        if np.sum(mean_in) > 0:
            distribution_a = mean_in / np.sum(mean_in)
        else:
            distribution_a = None
        word_a_index = np.random.choice(
            list(range(self.episode_op.graph.adj.shape[0])),
            size=1,
            p=distribution_a,
            replace=False,
        )[0]
        word_a_out = self.episode_op.graph.adj[word_a_index, :]

        if np.sum(word_a_out) > 0:
            distribution_b = word_a_out / np.sum(word_a_out)
        else:
            distribution_b = None
        word_b_index = np.random.choice(
            list(range(self.episode_op.graph.adj.shape[0])),
            size=1,
            p=distribution_b,
            replace=False,
        )[0]

        return [
            self.episode_op.graph.labels[word_a_index],
            self.episode_op.graph.labels[word_b_index],
        ]

    def mean_in_degree_word_pair(self):
        assert self.episode_op is not None
        mean_in = np.mean(self.episode_op.graph.adj, axis=0)
        if np.sum(mean_in) > 0:
            distribution_a = mean_in / np.sum(mean_in)
        else:
            distribution_a = None
        word_ix = np.random.choice(
            list(range(self.episode_op.graph.adj.shape[0])),
            size=2,
            p=distribution_a,
            replace=False,
        )
        return [
            self.episode_op.graph.labels[word_ix[0]],
            self.episode_op.graph.labels[word_ix[1]],
        ]

    def uniform_word_pair(self):
        assert self.episode_op is not None
        word_ix = np.random.choice(
            list(range(self.episode_op.graph.adj.shape[0])), size=2, replace=False
        )
        return [
            self.episode_op.graph.labels[word_ix[0]],
            self.episode_op.graph.labels[word_ix[1]],
        ]
