# Artificial Rabbits Optimization Algorithm + African Vultures

import numpy as np
from optimizer import Optimizer


class ARAVOA(Optimizer):

    def __init__(self, epoch=10000, pop_size=100, **kwargs):

        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):

        theta = 2 * (1 - (epoch+1)/self.epoch)
        ppp = (2 * np.random.rand() + 1) * (1 - epoch / self.epoch) + a
        _, best_list, _ = self.get_special_solutions(self.pop, best=2)
        pop_new = []
        for idx in range(0, self.pop_size):
            L = (np.exp(1) - np.exp((epoch / self.epoch)**2)) * (np.sin(2*np.pi*np.random.rand()))
            temp = np.zeros(self.problem.n_dims)
            rd_index = np.random.choice(np.arange(0, self.problem.n_dims), int(np.ceil(np.random.rand()*self.problem.n_dims)), replace=False)
            temp[rd_index] = 1
            R = L * temp        # Eq 2
            A = 2 * np.log(1.0 / np.random.rand()) * theta
            rand_idx = np.random.randint(0, self.pop_size)# Eq. 15
            if A > 1:
                pos_new = self.pop[rand_idx][self.ID_POS] + R * (self.pop[idx][self.ID_POS] - self.pop[rand_idx][self.ID_POS]) + \
                    np.round(0.5 * (0.05 + np.random.rand())) * np.random.normal(0, 1)      # Eq. 1
            # HYB - position is updated based on the Exploration phase of African Vultures
            # to improve the Artifical Rabbits
            elif A==1:
                F = ppp * (2 * np.random.rand() - 1)
                rand_pos = best_list[rand_idx][self.ID_POS]
                pos_new = rand_pos - (np.abs((2 * np.random.rand()) * rand_pos - self.pop[idx][self.ID_POS])) * F

            else:
                gr = np.zeros(self.problem.n_dims)
                rd_index = np.random.choice(np.arange(0, self.problem.n_dims), int(np.ceil(np.random.rand() * self.problem.n_dims)), replace=False)
                gr[rd_index] = 1        # Eq. 12
                H = np.random.normal(0, 1) * (epoch / self.epoch)       # Eq. 8
                b = self.pop[idx][self.ID_POS] + H * gr * self.pop[idx][self.ID_POS]        # Eq. 13
                pos_new = self.pop[idx][self.ID_POS] + R * (np.random.rand() * b - self.pop[idx][self.ID_POS])      # Eq. 11
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
