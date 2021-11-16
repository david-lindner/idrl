import numpy as np

from active_reward_learning.envs import Gridworld
from active_reward_learning.envs.gridworld import DOWN, LEFT, RIGHT, STAY, UP


def assert_deterministic_policy_and_get_action(probs):
    assert np.sum(probs) == 1
    assert np.sum(probs == 1) == 1
    a = np.argmax(probs)
    return a


def test_random_generation():
    env_123_1 = Gridworld(10, 10, 3, 2, 1000, 0.99, env_seed=123, wall_probability=0.2)
    env_123_2 = Gridworld(10, 10, 3, 2, 1000, 0.99, env_seed=123, wall_probability=0.2)
    env_345_1 = Gridworld(10, 10, 3, 2, 1000, 0.99, env_seed=345, wall_probability=0.2)
    env_345_2 = Gridworld(10, 10, 3, 2, 1000, 0.99, env_seed=345, wall_probability=0.2)
    initial_state_distribution_expected = np.zeros(100)
    initial_state_distribution_expected[env_123_1.initial_state] = 1
    assert np.all(
        env_123_1.initial_state_distribution == initial_state_distribution_expected
    )
    assert np.all(env_123_1.tiles == env_123_2.tiles)
    assert np.all(env_345_1.tiles == env_345_2.tiles)
    assert np.all(env_123_1.rewards == env_123_2.rewards)
    assert np.all(env_345_1.rewards == env_345_2.rewards)
    assert np.all(env_123_1.agent_xpos == env_123_2.agent_xpos)
    assert np.all(env_345_1.agent_xpos == env_345_2.agent_xpos)
    assert np.all(env_123_1.agent_ypos == env_123_2.agent_ypos)
    assert np.all(env_345_1.agent_ypos == env_345_2.agent_ypos)
    assert np.all(env_123_1.current_state == env_123_2.current_state)
    assert np.all(env_345_1.current_state == env_345_2.current_state)
    assert np.all(env_123_1.walls == env_123_2.walls)
    assert np.all(env_345_1.walls == env_345_2.walls)
    assert np.all(env_123_1.initial_state == env_123_2.initial_state)
    assert np.all(env_345_1.initial_state == env_345_2.initial_state)
    assert np.all(
        env_123_1.initial_state_distribution == env_123_2.initial_state_distribution
    )
    assert np.all(
        env_345_1.initial_state_distribution == env_345_2.initial_state_distribution
    )
    assert np.any(env_123_1.tiles != env_345_1.tiles)
    assert np.any(env_123_1.tiles != env_345_2.tiles)
    assert np.any(env_123_2.tiles != env_345_1.tiles)
    assert np.any(env_123_2.tiles != env_345_2.tiles)
    assert np.any(env_123_1.rewards != env_345_1.rewards)
    assert np.any(env_123_1.rewards != env_345_2.rewards)
    assert np.any(env_123_2.rewards != env_345_1.rewards)
    assert np.any(env_123_2.rewards != env_345_2.rewards)
    assert np.any(env_123_1.walls != env_345_1.walls)
    assert np.any(env_123_1.walls != env_345_2.walls)
    assert np.any(env_123_2.walls != env_345_1.walls)
    assert np.any(env_123_2.walls != env_345_2.walls)
    assert np.any(env_123_1.initial_state != env_345_1.initial_state)
    assert np.any(env_123_1.initial_state != env_345_2.initial_state)
    assert np.any(env_123_2.initial_state != env_345_1.initial_state)
    assert np.any(env_123_2.initial_state != env_345_2.initial_state)
    assert np.any(
        env_123_1.initial_state_distribution != env_345_1.initial_state_distribution
    )
    assert np.any(
        env_123_1.initial_state_distribution != env_345_2.initial_state_distribution
    )
    assert np.any(
        env_123_2.initial_state_distribution != env_345_1.initial_state_distribution
    )
    assert np.any(
        env_123_2.initial_state_distribution != env_345_2.initial_state_distribution
    )


def test_state_representation():
    env = Gridworld(10, 15, 3, 2, 1000, 0.99, env_seed=123)
    assert env._get_state_from_agent_pos(2, 2) == 22
    assert env._get_state_from_agent_pos(2, 3) == 32
    assert env._get_state_from_agent_pos(3, 2) == 23
    assert env._get_state_from_agent_pos(0, 0) == 0
    assert env._get_agent_pos_from_state(22) == (2, 2)
    assert env._get_agent_pos_from_state(32) == (2, 3)
    assert env._get_agent_pos_from_state(23) == (3, 2)
    assert env._get_agent_pos_from_state(0) == (0, 0)
    env = Gridworld(15, 10, 3, 2, 1000, 0.99, env_seed=123)
    assert env._get_state_from_agent_pos(2, 2) == 32
    assert env._get_state_from_agent_pos(2, 3) == 47
    assert env._get_state_from_agent_pos(3, 2) == 33
    assert env._get_state_from_agent_pos(0, 0) == 0
    assert env._get_agent_pos_from_state(32) == (2, 2)
    assert env._get_agent_pos_from_state(47) == (2, 3)
    assert env._get_agent_pos_from_state(33) == (3, 2)
    assert env._get_agent_pos_from_state(0) == (0, 0)


def test_lp_solver_goes_to_highest_reward_point():
    np.random.seed(1)
    seeds = np.random.randint(1, 2000, 10)
    for seed in seeds:
        env = Gridworld(10, 15, 3, 2, 1000, 0.99999, env_seed=seed, wall_probability=0)
        opt_policy = env.get_lp_solution()
        s = env.reset()
        done = False
        while not done:
            a = assert_deterministic_policy_and_get_action(opt_policy.matrix[s])
            s, reward, done, info = env.step(a)
        assert np.isclose(reward, max(env.object_rewards))


def test_actions():
    actions = (
        [LEFT, UP, UP, DOWN, DOWN, RIGHT, RIGHT, LEFT, DOWN, STAY, RIGHT]
        + [RIGHT] * 15
        + [DOWN] * 15
    )
    x_list_expected = (
        [0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 2, 3] + list(range(4, 15)) + [14] * 3 + [14] * 15
    )
    y_list_expected = (
        [0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 3] + [3] * 15 + list(range(4, 10)) + [9] * 9
    )
    np.random.seed(3)
    seeds = np.random.randint(1, 10000, 3)
    for seed in seeds:
        env = Gridworld(15, 10, 3, 2, 1000, 0.99, env_seed=seed)
        env.agent_xpos = 0
        env.agent_ypos = 0
        env.initial_state = env._get_state_from_agent_pos(
            env.agent_xpos, env.agent_ypos
        )
        for _ in range(3):
            state = env.reset()
            assert env.agent_xpos == 0
            assert env.agent_ypos == 0
            assert env.current_state == 0
            assert state == 0
            for i, a in enumerate(actions):
                state, reward, done, info = env.step(a)
                assert env.agent_xpos == x_list_expected[i]
                assert env.agent_ypos == y_list_expected[i]
                assert state == env._get_state_from_agent_pos(
                    x_list_expected[i], y_list_expected[i]
                )


def test_custom_point_gridworld():
    tiles = np.array(
        [
            [4, 0, 2, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 3, 4, 3, 0],
        ],
        dtype=np.int,
    )
    object_rewards = [0.1, 0.2, 0.3, 0.4]
    object_radii = [0, 0, 0, 0]
    xpos, ypos = 2, 2

    actions = [UP, LEFT, RIGHT, RIGHT, DOWN, DOWN, LEFT, LEFT, DOWN, RIGHT, RIGHT]
    x_list_expected = [2, 1, 2, 3, 3, 3, 2, 1, 1, 2, 3]
    y_list_expected = [1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4]
    rewards_expected = [0, 0.1, 0, 0.1, 0, 0, 0.2, 0, 0.3, 0.4, 0.3]

    np.random.seed(9)
    seeds = np.random.randint(1, 10000, 3)
    for seed in seeds:
        env = Gridworld(
            5,
            5,
            4,
            2,
            1000,
            0.99,
            tiles=tiles,
            object_rewards=object_rewards,
            object_radii=object_radii,
            agent_xpos=xpos,
            agent_ypos=ypos,
        )
        for _ in range(3):
            env.reset()
            assert env.agent_xpos == 2
            assert env.agent_ypos == 2
            for i, a in enumerate(actions):
                state, reward, done, info = env.step(a)
                assert env.agent_xpos == x_list_expected[i]
                assert env.agent_ypos == y_list_expected[i]
                assert state == env._get_state_from_agent_pos(
                    x_list_expected[i], y_list_expected[i]
                )
                assert np.allclose(reward, rewards_expected[i])

            env.reset()
            opt_policy = env.get_lp_solution()
            s = env.reset()

            a = assert_deterministic_policy_and_get_action(opt_policy.matrix[s])
            s, reward, done, info = env.step(a)
            assert a == DOWN
            assert np.allclose(reward, 0.2)

            a = assert_deterministic_policy_and_get_action(opt_policy.matrix[s])
            s, reward, done, info = env.step(a)
            assert a == DOWN
            assert np.allclose(reward, 0.4)

            done = False
            while not done:
                a = assert_deterministic_policy_and_get_action(opt_policy.matrix[s])
                s, reward, done, info = env.step(a)
                assert a == STAY
                assert np.allclose(reward, 0.4)


def test_candidate_policies():
    np.random.seed(99)
    seeds = np.random.randint(1, 200000, 3)
    for seed in seeds:
        env = Gridworld(15, 10, 3, 2, 1000, 0.99, env_seed=seed, random_action_prob=0)
        candidate_policies = env.get_candidate_policies()
        for target, policy in zip(env.object_states, candidate_policies):
            x_target, y_target = env._get_agent_pos_from_state(target)
            for start in np.random.choice(np.arange(env.N_states), 5):
                x_start, y_start = env._get_agent_pos_from_state(start)
                l1_dist = abs(x_start - x_target) + abs(y_start - y_target)
                env.initial_state = start
                s = env.reset()
                assert s == start
                n = 0
                done = False
                while not done:
                    a = assert_deterministic_policy_and_get_action(policy.matrix[s])
                    s, reward, done, info = env.step(a)
                    n += 1
                    assert (n < l1_dist and s != target) or s == target


def test_optimal_policy_in_candidate_policies():
    seeds = np.random.randint(1, 200000, 3)
    for seed in seeds:
        env = Gridworld(
            15,
            12,
            10,
            10,
            1000,
            0.99,
            env_seed=seed,
            random_action_prob=0.2,
            gaussian_peaks_as_rewards=True,
            add_optimal_policy_to_candidates=True,
        )
        candidate_policies = env.get_candidate_policies()
        optimal_policy = env.get_lp_solution()
        optimal_policy_return = optimal_policy.evaluate(env, rollout=False)
        optimal_policy_in_candidates = False
        for policy in candidate_policies:
            policy_return = policy.evaluate(env, rollout=False)
            if optimal_policy_return == policy_return:
                optimal_policy_in_candidates = True
                break
        assert optimal_policy_in_candidates


def test_gridworld_repr():
    np.random.seed(11121)
    seeds = np.random.randint(1, 20000, 10)
    for seed in seeds:
        env = Gridworld(
            20, 15, 20, 4, 1000, 0.99999, env_seed=seed, wall_probability=0.3
        )
        all_repr = env.get_all_states_repr()
        N = 100
        for _ in range(N):
            s1 = np.random.randint(0, env.N_states - 1)
            s2 = np.random.randint(0, env.N_states - 1)
            x1, y1 = env._get_agent_pos_from_state(s1)
            x2, y2 = env._get_agent_pos_from_state(s2)
            repr1 = env.get_state_repr(s1)
            repr2 = env.get_state_repr(s2)
            assert repr1.shape == (env.Ndim_repr,)
            assert repr2.shape == (env.Ndim_repr,)
            assert np.all(np.logical_or(repr1 == 0, repr1 == 1))
            assert np.all(np.logical_or(repr2 == 0, repr2 == 1))
            assert np.all(repr1 == all_repr[s1])
            assert np.all(repr2 == all_repr[s2])
            if env.tiles[y1, x1] == env.tiles[y2, x2]:
                assert np.all(repr1 == repr2)
            else:
                assert np.any(repr1 != repr2)


def transitions_for_random_action_prob():
    np.random.seed(12453)
    random_action_prob

    for _ in range(10):
        wall_probability = np.random.random()
        random_action_prob = np.random.random()
        env = Gridworld(
            10,
            15,
            20,
            4,
            1000,
            0.99999,
            env_seed=seed,
            wall_probability=wall_probability,
            random_action_prob=random_action_prob,
        )

        assert (
            env.transitions[LEFT, 3, 2]
            == 1 - random_action_prob + random_action_prob / 5
        )
        assert (
            env.transitions[RIGHT, 2, 3]
            == 1 - random_action_prob + random_action_prob / 5
        )
        assert (
            env.transitions[UP, 13, 3]
            == 1 - random_action_prob + random_action_prob / 5
        )
        assert (
            env.transitions[DOWN, 3, 13]
            == 1 - random_action_prob + random_action_prob / 5
        )
        assert (
            env.transitions[STAY, 5, 5]
            == 1 - random_action_prob + random_action_prob / 5
        )

    env = Gridworld(
        10,
        15,
        20,
        4,
        1000,
        0.99999,
        env_seed=seed,
        wall_probability=0.2,
        random_action_prob=0,
    )
    assert np.all(np.logical_or(env.transitions == 0, env.transitions == 1))

    env = Gridworld(
        10,
        15,
        20,
        4,
        1000,
        0.99999,
        env_seed=seed,
        wall_probability=0.2,
        random_action_prob=1,
    )

    assert np.all(env.transitions[LEFT] == env.transitions[RIGHT])
    assert np.all(env.transitions[RIGHT] == env.transitions[UP])
    assert np.all(env.transitions[UP] == env.transitions[DOWN])
    assert np.all(env.transitions[DOWN] == env.transitions[STAY])
    assert np.all(env.transitions[STAY] == env.transitions[LEFT])


def test_gaussian_reward_peaks():
    np.random.seed(13252)
    seeds = np.random.randint(1, 20000, 10)
    for seed in seeds:
        env = Gridworld(
            20,
            15,
            20,
            4,
            1000,
            0.99999,
            env_seed=seed,
            wall_probability=0.3,
            gaussian_peaks_as_rewards=True,
        )
        assert np.all(env.rewards != 0)


def test_custom_gaussian_gridworld():
    tiles = np.array(
        [
            [2, 0, 2, 0, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int,
    )
    object_rewards = [-1, 1]
    object_radii = [0.5, 0.5]
    xpos, ypos = 2, 2

    env = Gridworld(
        5,
        5,
        2,
        2,
        1000,
        0.99,
        tiles=tiles,
        object_rewards=object_rewards,
        object_radii=object_radii,
        agent_xpos=xpos,
        agent_ypos=ypos,
    )
    R = env.rewards
    assert R[env._get_state_from_agent_pos(0, 0)] > 1
    assert R[env._get_state_from_agent_pos(2, 0)] > 1
    assert R[env._get_state_from_agent_pos(4, 0)] > 1
    assert (
        R[env._get_state_from_agent_pos(2, 0)] > R[env._get_state_from_agent_pos(0, 0)]
    )
    assert (
        R[env._get_state_from_agent_pos(2, 0)] > R[env._get_state_from_agent_pos(4, 0)]
    )
    assert (
        R[env._get_state_from_agent_pos(1, 0)] < R[env._get_state_from_agent_pos(0, 0)]
    )
    assert (
        R[env._get_state_from_agent_pos(1, 0)] < R[env._get_state_from_agent_pos(2, 0)]
    )
    assert (
        R[env._get_state_from_agent_pos(3, 0)] < R[env._get_state_from_agent_pos(2, 0)]
    )
    assert (
        R[env._get_state_from_agent_pos(3, 0)] < R[env._get_state_from_agent_pos(4, 0)]
    )
    assert R[env._get_state_from_agent_pos(1, 3)] < -1
    assert R[env._get_state_from_agent_pos(3, 3)] < -1
    assert (
        R[env._get_state_from_agent_pos(0, 3)] > R[env._get_state_from_agent_pos(1, 3)]
    )
    assert (
        R[env._get_state_from_agent_pos(2, 3)] > R[env._get_state_from_agent_pos(1, 3)]
    )
    assert (
        R[env._get_state_from_agent_pos(2, 3)] > R[env._get_state_from_agent_pos(3, 3)]
    )
    assert (
        R[env._get_state_from_agent_pos(4, 3)] > R[env._get_state_from_agent_pos(3, 3)]
    )


def test_reachable_tiles():
    np.random.seed(1)
    seeds = np.random.randint(1, 2000, 10)
    for seed in seeds:
        wall_prob = np.random.random() / 2
        env = Gridworld(
            10, 15, 3, 2, 1000, 0.99999, env_seed=seed, wall_probability=wall_prob
        )
        states_visited = set()
        for _ in range(1000):
            a = np.random.randint(0, 4)
            s, reward, done, info = env.step(a)
            states_visited.add(str(s))
        assert env.get_number_of_reachable_tiles() >= len(states_visited)
