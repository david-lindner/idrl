import itertools
import time
from typing import Dict, List, Optional, Tuple, Union

import cvxopt
import gym
import matplotlib.axes
import numpy as np
import scipy.sparse as sp

from active_reward_learning.common.policy import TabularPolicy
from active_reward_learning.util.helpers import get_deterministic_policy_matrix

cvxopt.solvers.options["show_progress"] = False
EPS = np.finfo(np.float32).eps


class TabularMDP(gym.Env):
    """
    Implements a simple tabular MDP with discrete state and action spaces.

    Provides a standard gym interface and some additional helper methods
    that allow to deal with the transition matrix, value functions and get
    the optimal policy (based on a LP solver).

    - The states are labeled with integers and the rewards are a 1D-array.
    - The rewards only depend on the state that the agent ends up in after taking
      an action, i.e. R(s, a, s') = R(s').
    - Currently only a fixed episode length is supported.

    Attributes
    ----------------
    N_states (int): number of states
    N_actions (int): number of actions
    rewards (np.ndarray): 1D-array of rewards
    transitions (np.ndarray): transition proability matrix with dimensions
                              (N_actions, N_states, N_states)
    discount_factor (float): discount factor (gamma) for the return
    initial_state (int): if not None, the agent will always start in this state
                         after resetting the environment. Otherwise, the agent
                         will always start in a random state
    terminal_states (list): if the agent is in one of these states, the episode
                            ends automatically
    episode_length (int): after how many steps an episode ends
    state_space (gym.spaces.Discrete): gym state space
    action_space (gym.spaces.Discrete): gym action space
    current_state (int): state the agent is in currently
    """

    def __init__(
        self,
        N_states: int,
        N_actions: int,
        rewards: np.ndarray,
        transitions: Optional[np.ndarray],
        discount_factor: float,
        terminal_states: List[int],
        episode_length: Optional[int],
        initial_state: Optional[int],
        use_sparse_transitions: bool = False,
        sparse_transitions: Optional[List[sp.spmatrix]] = None,
        observation_noise: float = 0,
        observation_type: str = "state",
    ):
        # rewards only state dependant for now
        assert rewards.shape == (N_states,)
        if transitions is not None:
            assert transitions.shape == (N_actions, N_states, N_states)
        assert initial_state is None or (
            initial_state >= 0 and initial_state < N_states
        )
        assert len(terminal_states) > 0 or episode_length and episode_length > 0
        assert N_states % 1 == 0 and N_actions % 1 == 0
        assert transitions is not None or (
            use_sparse_transitions and sparse_transitions is not None
        )
        assert observation_type in ("state", "raw", "features")
        self.N_states = int(N_states)
        self.N_actions = int(N_actions)
        self.rewards = rewards
        self.transitions = transitions
        self.discount_factor = discount_factor
        self.observation_noise = observation_noise
        self.observation_type = observation_type
        self.initial_state = initial_state
        if initial_state is None:
            self.initial_state_distribution = np.ones(N_states) / N_states
        else:
            self.initial_state_distribution = np.zeros(N_states)
            self.initial_state_distribution[initial_state] = 1
        self.terminal_states = terminal_states
        self.episode_length = episode_length
        self.state_space = gym.spaces.Discrete(N_states)
        self.action_space = gym.spaces.Discrete(N_actions)
        self.current_state = 0
        self.value_trafo_dict: Dict[str, np.ndarray] = dict()
        self.return_trafo_dict: Dict[str, np.ndarray] = dict()
        self.transition_matrix_dict: Dict[str, np.ndarray] = dict()
        self._all_states_repr: Optional[List[np.ndarray]] = None
        self._is_deterministic: Optional[bool] = None
        self.use_sparse_transitions = use_sparse_transitions
        if self.use_sparse_transitions:
            self.sparse_transitions: List[sp.spmatrix] = []
            if sparse_transitions is not None:
                self.sparse_transitions = sparse_transitions
            else:
                for i in range(self.N_actions):
                    self.sparse_transitions.append(
                        sp.csr_matrix(self.transitions[i, :, :])
                    )

        if self.initial_state is None:
            self.current_state = np.random.choice(self.N_states)
        else:
            self.current_state = self.initial_state
        self.timestep = 0
        self.solve_time: float = 0
        self.evaluate_time: float = 0

        if self.observation_type == "raw":
            self.observation_space = gym.spaces.Box(
                0, 1, shape=(self.N_states,), dtype=np.uint8
            )
        elif self.observation_type == "state":
            self.observation_space = gym.spaces.Discrete(self.N_states)

    def reset(self) -> Union[int, np.ndarray]:
        """
        Reset the environment
        """
        if self.initial_state is None:
            self.current_state = np.random.choice(self.N_states)
        else:
            self.current_state = self.initial_state
        self.timestep = 0
        return self.get_observation(self.current_state)

    def step(self, action: int) -> Tuple[Union[int, np.ndarray], float, bool, Dict]:
        """
        Take one action, return the new state and reward and update the environment.
        """
        assert action % 1 == 0
        action = int(action)
        self.timestep += 1
        if self.use_sparse_transitions:
            next_state_dist = self.sparse_transitions[action][
                self.current_state
            ].toarray()[0]
        else:
            next_state_dist = self.transitions[action, self.current_state]
        self.current_state = np.random.choice(self.N_states, p=next_state_dist)
        reward = self.rewards[self.current_state]
        done = self.current_state in self.terminal_states or (
            self.episode_length is not None and self.timestep >= self.episode_length
        )
        info: Dict = dict()
        info["state"] = self.current_state
        info["gp_repr"] = self.get_state_repr(self.current_state)
        return self.get_observation(self.current_state), reward, done, info

    def render(
        self, mode: str = "human", close: bool = False
    ) -> Union[str, np.ndarray, None]:
        """
        Render the MDP (only implements ansi mode)
        """
        if mode == "ansi":
            repr: List[str] = []
            for i, r in enumerate(self.rewards):
                if self.current_state == i:
                    repr.append("X")
                else:
                    repr.append("{:.2f}".format(r))
            return "  ".join(repr)
        elif mode == "human":
            raise NotImplementedError()
        elif mode == "rgb_array":
            raise NotImplementedError()
        else:
            raise NotImplementedError("Unsupported mode '{}'".format(mode))

    def get_reward(self, state: int) -> float:
        """ Query an individual reward. """
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise)
        else:
            noise = 0
        return self.rewards[state] + noise

    def get_transition_matrix_for_policy(
        self, policy: TabularPolicy, sparse: bool = False
    ) -> Union[np.ndarray, sp.spmatrix]:
        """
        Get the transition matrix that is induced by a policy, i.e the state
        transition probabilites when commiting to one specific policy.

        Args:
        ----------
        policy (TabularPolicy)
        sparse (bool): use the sparse transition matrix (if True a sparse matrix
                       will be returned)

        Returns:
        ---------
            (sparse) matrix of shape (N_states, N_states)
        """
        assert policy.matrix.shape == (self.N_states, self.N_actions)
        assert policy.matrix.dtype == np.float32
        policy_str = str(hash(policy.matrix.tostring()))
        if sparse:
            policy_str += "sparse"
        if policy_str in self.transition_matrix_dict:
            return self.transition_matrix_dict[policy_str]
        else:
            if sparse or self.transitions is None:
                T_pi = sp.vstack(
                    [
                        sum(
                            policy.matrix[s, a] * self.sparse_transitions[a][s, :]
                            for a in range(self.N_actions)
                        )
                        for s in range(self.N_states)
                    ]
                )
            else:
                T_pi = np.stack(
                    [
                        sum(
                            policy.matrix[s, a] * self.transitions[a][s, :]
                            for a in range(self.N_actions)
                        )
                        for s in range(self.N_states)
                    ]
                )
            if not sparse and self.transitions is None:
                T_pi = T_pi.toarray()
            self.transition_matrix_dict[policy_str] = T_pi
            return T_pi

    def get_value_trafo_for_policy(
        self, policy: TabularPolicy, sparse: bool = False
    ) -> Union[np.ndarray, sp.spmatrix]:
        """
        Return the matrix
            M = (I-\\gamma T_\\pi)^{-1}
        which can be used to calculate the value of a policy.

        Uses caching in a dictionary to avoid duplicate computations.

        Args:
        --------------
        policy (TabularPolicy)
        sparse (bool): use the sparse transition matrix (if True a sparse matrix
                       will be returned)

        Returns:
        --------------
            (sparse) matrix of shape (N_states, N_states)
        """
        policy_str = str(hash(policy.matrix.tostring()))
        if sparse:
            policy_str += "sparse"
        if policy_str in self.value_trafo_dict:
            return self.value_trafo_dict[policy_str]
        else:
            N = self.N_states
            T_pi = self.get_transition_matrix_for_policy(
                policy, sparse=(sparse or self.transitions is None)
            )
            if sparse or self.transitions is None:
                A = sp.identity(self.N_states, format="csc") - T_pi.multiply(
                    self.discount_factor
                )
                M = sp.linalg.inv(A).toarray()
            else:
                M = np.linalg.inv(np.identity(N) - self.discount_factor * T_pi)
            self.value_trafo_dict[policy_str] = M
            return M

    def get_return_trafo_for_policy(
        self, policy: TabularPolicy, sparse: bool = False
    ) -> Union[np.ndarray, sp.spmatrix]:
        """
        Return the vector
            W = q^T (I-\\gamma T_\\pi)^{-1}
        which can be used to calculate the return of a policy.

        Uses caching in a dictionary to avoid duplicate computations.

        Args:
        --------------
        policy (TabularPolicy)

        Returns:
        --------------
        W (np.ndarray): vector of shape N_states
        """
        policy_str = str(hash(policy.matrix.tostring()))
        if policy_str in self.return_trafo_dict:
            return self.return_trafo_dict[policy_str]
        else:
            M = self.get_value_trafo_for_policy(policy, sparse=sparse)
            if sparse:
                W = sp.csr_matrix(self.initial_state_distribution).dot(M)
            else:
                W = np.dot(self.initial_state_distribution, M)
            self.return_trafo_dict[policy_str] = W
            return W

    def get_value_function(
        self,
        policy: TabularPolicy,
        rewards: Optional[np.ndarray] = None,
        sparse: bool = False,
    ) -> Union[np.ndarray, sp.spmatrix]:
        """
        Get the value function of a specific policy.

        The true reward can be overwritten with a different vector of rewards.

        Args:
        ---------
        policy (TabularPolicy)
        rewards (np.ndarray): 1D-array of rewards (if not set the true rewards
                              are used)

        Returns:
        ---------
        np.ndarray of length N_states containing V^\\pi(s) for all states
        """
        if rewards is None:
            rewards = self.rewards
        M = self.get_value_trafo_for_policy(policy, sparse=sparse)
        T_pi = self.get_transition_matrix_for_policy(policy, sparse=sparse)
        if sparse:
            T_pi = T_pi.toarray()
        # additionally multiply with T_pi here, because the rewards are defined by resulting states
        V = np.dot(M, np.dot(T_pi, rewards))
        return V

    def evaluate_policy(
        self,
        policy: TabularPolicy,
        initial_state_distribution: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculates the expected return of a given policy.

        Assumes a initial state distribution \\mu to be given, it then returns
            \\mathbb{E}G^\\pi = \\sum_{s_i} \\mu_i \\cdot V^\\pi(s_i)
        If the state distribution is not given, a uniform distribution is assumed.

        Args:
        ----------
        policy (TabularPolicy)
        initial_state_distribution (np.ndarray): array of length N_states that
                                                 contains the probability of
                                                 starting in each state

        Returns:
        ----------
        float: expected return of the policy
        """
        t = time.time()
        if initial_state_distribution is None:
            initial_state_distribution = self.initial_state_distribution
        assert policy.matrix.shape == (self.N_states, self.N_actions)
        assert np.allclose(np.sum(initial_state_distribution), 1)
        V = self.get_value_function(
            policy, sparse=self.use_sparse_transitions, rewards=rewards
        )
        assert V.shape == initial_state_distribution.shape
        self.evaluate_time += time.time() - t
        policy_return = np.dot(initial_state_distribution, V)
        return policy_return

    def get_greedy_policy_for_value_function(
        self, V: np.ndarray, rewards: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get the policy that acts greedily w.r.t. a specific value function.

        Args:
        ---------
        V (np.ndarray): 1D-array of length N_states containing a value function
        rewards (np.ndarray): optionally array of rewards (if not set the true
                              rewards are used instead)

        Returns:
        --------
        np.ndarray: 2D-array of shape (N_states, N_actions() containing greedy
                    actions w.r.t. V and rewards (in one-hot encoding)
        """
        assert V.shape == (self.N_states,)
        if rewards is None:
            rewards = self.rewards

        policy = np.zeros((self.N_states, self.N_actions), dtype=np.float32)
        for i in range(self.N_states):
            greedy_a = self.N_actions - 1
            if self.use_sparse_transitions:
                max_val = self.sparse_transitions[greedy_a][i].dot(
                    rewards + self.discount_factor * V
                )
            else:
                max_val = np.dot(
                    self.transitions[greedy_a, i], rewards + self.discount_factor * V
                )
            for a in range(self.N_actions - 1):
                if self.use_sparse_transitions:
                    val = self.sparse_transitions[a][i].dot(
                        rewards + self.discount_factor * V
                    )
                else:
                    val = np.dot(
                        self.transitions[a, i], rewards + self.discount_factor * V
                    )
                if val > max_val:
                    greedy_a = a
                    max_val = val
            policy[i, greedy_a] = 1
        return TabularPolicy(policy)

    def get_lp_solution(
        self, rewards: Optional[np.ndarray] = None, return_value: bool = False
    ) -> TabularPolicy:
        """
        Get the linear programming solution of the MDP using cvxopt.

        Solves the following LP:
            find V_i
            minimize 1/N sum_i V_i
            s.t.
                sum_j (T[s_i, a_k, s_j] * (r_i + gamma * V_j)) <= V_i
                    for all states s_i and actions a_k
        (see eg. https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/mdps-exact-methods.pdf)

        Args:
        ---------
        rewards (np.ndarray): 1D-array of rewards (if not set the true rewards
                              are used)
        return_value (bool): if set to True, the optimal value function will
                             be returned; otherwise the optimal poilcy will be
                             returned

        Returns:
        --------
        np.ndarray: TabularPolicy
                    or 1D-array with a value function (depending on the value of `return_value`)
        """
        t = time.time()
        if rewards is None:
            rewards = self.rewards
        c = cvxopt.matrix(np.full(self.N_states, 1 / self.N_states))

        if self.use_sparse_transitions:
            assert self.sparse_transitions is not None
            b = cvxopt.matrix(
                -np.reshape(sp.vstack(self.sparse_transitions).dot(rewards), (-1, 1))
            )
            A = sp.vstack(self.sparse_transitions).multiply(
                self.discount_factor
            ) - sp.vstack([sp.identity(self.N_states)] * self.N_actions)
            A = A.tocoo()
            A = cvxopt.spmatrix(A.data, A.row, A.col)
        else:
            reshaped_transitions = np.reshape(
                self.transitions, (self.N_actions * self.N_states, -1)
            )
            b = cvxopt.matrix(
                -np.reshape(np.dot(reshaped_transitions, rewards), (-1, 1))
            )
            A = cvxopt.matrix(
                self.discount_factor * reshaped_transitions
                - np.tile(np.identity(self.N_states), (self.N_actions, 1))
            )

        sol = cvxopt.solvers.lp(c, A, b)
        V = sol["x"]
        V = np.array(V)[:, 0]
        self.solve_time += time.time() - t
        if return_value:
            return V
        else:
            return self.get_greedy_policy_for_value_function(V, rewards=rewards)

    def get_policy_iteration_solution(
        self,
        max_it,
        rewards: Optional[np.ndarray] = None,
        return_value: bool = False,
        init_pi=None,
    ) -> TabularPolicy:
        """
        Policy iteration algorithm.

        Parameters
        ----------
        max_it: int
            Max number of iteration to run PI for.
        init_pi: np.array
            Deterministic policy (S dimensional array) to initialize the
            algorithm with. If None we initialize with the policy that is
            greedy wrt self.R.

        Returns
        -------
        pi: TabularPolicy
            Optimal policy.
        V: np.array
            Optimal value function.
        """
        t = time.time()
        if rewards is None:
            rewards = self.rewards
        # Init
        if init_pi is None:
            pi = self.get_greedy_policy_for_value_function(rewards, rewards=rewards)
        else:
            pi = init_pi

        # Actual PI
        for i in range(max_it):
            pi_old = np.copy(pi)
            V = self.get_value_function(pi, rewards=rewards)
            pi = self.get_greedy_policy_for_value_function(V, rewards=rewards)
            if np.all(pi_old == pi):
                break

        self.solve_time += time.time() - t
        if return_value:
            return V
        else:
            return pi

    def get_candidate_policies(self) -> List[TabularPolicy]:
        """
        Generate a set of candidate policies.

        This set is meant to be used to test algorithms that query
        rewards to find the best policy in a restricted policy space.

        This function creates all possible policies (removing duplicates
        that lead to the exact same transition matrix). It is supposed
        to be overwritten for other environments.
        """
        assert not self.use_sparse_transitions
        relevant_actions_per_state = []
        for s in range(self.N_states):
            relevant_actions = [0]
            old_transitions = self.transitions[0, s, :]
            for a in range(1, self.N_actions):
                new_transitions = self.transitions[a, s, :]
                if np.any(new_transitions != old_transitions):
                    relevant_actions.append(a)
                old_transitions = new_transitions
            relevant_actions_per_state.append(relevant_actions)
        policies = [
            TabularPolicy(
                get_deterministic_policy_matrix(
                    np.array(policy, dtype=np.int), self.N_actions
                )
            )
            for policy in itertools.product(*relevant_actions_per_state)
        ]
        return policies

    def visualize_reward_estimate(
        self, mu: np.ndarray, sigma: np.ndarray, ax: matplotlib.axes.Axes
    ) -> None:
        """
        Plot a reward estimate consisting of means and variances to a pyplot axes.

        Shows the state space one-dimensional on the x-axis and the mean and variances as a line
        and a shaded region on the y-axis. For comparison, the true reward is plotted as a black line.

        Args:
        -----------
        mu (np.ndarray): 1d-array of mean prediction of the model for each state
        sigma (np.ndarray): 1d-array of prediction variances of the model for each state
        ax (matplotlib.axes.Axes): axes to plot to
        """
        assert mu.shape == (self.N_states,)
        assert sigma.shape == (self.N_states,)
        states = np.arange(self.N_states)
        ax.plot(states, self.rewards, color="black", label="true")
        ax.plot(states, mu, color="orange", label="prediction")
        ax.fill_between(states, mu - sigma, mu + sigma, alpha=0.3, color="orange")

    def visualize_value(
        self,
        value: np.ndarray,
        ax: matplotlib.axes.Axes,
        highlight_states: Optional[List[int]] = None,
    ) -> None:
        """
        Plot a value function to a pyplot axes.

        Shows the state space one-dimensional on the x-axis and the value function on the y-axis.

        Args:
        -----------
        value (np.ndarray): 1d-array containing the value of each state
        ax (matplotlib.axes.Axes): axes to plot to
        highlight_state (int): if given this specific state will be hightlighted
        """
        assert value.shape == (self.N_states,)
        ax.plot(np.arange(self.N_states), value)
        if highlight_states is not None:
            for highlight_state in highlight_states:
                ax.axvline(highlight_state)

    def visualize_policy(self, policy: TabularPolicy, ax: matplotlib.axes.Axes) -> None:
        """
        Plot a policy to a pyplot axes.

        Shows the state space one-dimensional on the x-axis and the policy encoding on the y-axis.

        Args:
        -----------
        policy (TabularPolicy)
        ax (matplotlib.axes.Axes): axes to plot to
        """
        deterministic_policy = np.argmax(policy.matrix, axis=1)
        ax.plot(np.arange(self.N_states), deterministic_policy)

    def get_state_repr(self, state: int) -> np.ndarray:
        """
        This can be implemented to define vector representations of the states.

        These can e.g. be used to define GP kernels.
        """
        return self.get_observation(state)

    def get_all_states_repr(self) -> List[np.ndarray]:
        """
        Return a list that contains the vector representation of each state.
        """
        if self._all_states_repr is None:
            self._all_states_repr = []
            for state in range(self.N_states):
                state_repr = self.get_state_repr(state)
                self._all_states_repr.append(state_repr)
        return self._all_states_repr

    def is_deterministic(self) -> bool:
        """
        Return True if the state transitions are deterministic, i.e. if the
        transition matrix only contains 0s and 1s.
        """
        if self._is_deterministic is None:
            self._is_deterministic = True
            for a in range(self.N_actions):
                for s in range(self.N_states):
                    if self.use_sparse_transitions:
                        if np.sum(self.sparse_transitions[a][s].toarray() == 1) != 1:
                            self._is_deterministic = False
                            return False
                    else:
                        if np.sum(self.transitions[a, s] == 1) != 1:
                            self._is_deterministic = False
                            return False
            return True
        else:
            return self._is_deterministic

    def get_observation(self, state: int) -> np.ndarray:
        if self.observation_type == "raw":
            return self._get_raw_observation(state)
        elif self.observation_type == "features":
            return self._get_feature_observation(state)
        else:
            return np.atleast_1d(np.array(state))

    def _get_raw_observation(self, state: int) -> np.ndarray:
        return np.arange(self.N_states) == state

    def _get_feature_observation(self, state: int) -> np.ndarray:
        raise NotImplementedError()

    def set_feature_function(self, feature_function):
        self._get_feature_observation = feature_function
