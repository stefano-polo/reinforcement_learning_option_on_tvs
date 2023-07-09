import sys

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from numpy import log, sqrt

sys.path.insert(1, "../pricing")

from closedforms import BS_European_option_closed_form
from envs.utils import build_allocation_time_grid, sign_renormalization
from read_market import LoadFromTxt
from targetvol import (
    CholeskyTDependent,
    Drift,
    Markowitz_solution,
    Strategy,
    TVSForwardCurve,
    optimization_only_long,
)

from pricing import (
    DiscountingCurve,
    EquityForwardCurve,
    ForwardVariance,
    LV_model,
    quad_piecewise,
)


class TVS_LV_ENV_reward2(gym.Env):
    """Target volatility strategy Option environment with a Local volatility model for the assets"""

    def __init__(
        self,
        market_folder: str = "../market_data/FakeSmilesDisplacedDiffusion",
        asset_names: list or tuple = ["DJ 50 EURO E", "S&P 500 NET EUR"],
        allocation_frequency: str = "monthly",
        target_volatility: float = 5 / 100,
        tvs_spot_value: float = 1.0,
        option_strike: float = 1.0,
        option_maturity: float = 2.0,
        neural_network_action_parameterization: str = None,
        action_constraints: str = None,
        action_bound: float = 5.0,
        overall_long_position: float = None,
        overall_short_position: float = None,
        n_sim_for_cache: int = int(2e3),
    ) -> None:
        """
        Target volatility strategy Option RL environment with a Local volatility model for the assets. The environment implements
        the second reward function experimented in the paper. In this environment the action space and the observation space are
        continuous. The observation state is a vector of size (n_assets + 2) with the following structure: [stock price, tvs level, time]
        and the state space is the following block: lower_bound = [-2.5, 0., 0.]; upper_bound = [2.5, 10 * tvs_spot_value, maturity].
        :param market_folder (str, default="../market_data/FakeSmilesDisplacedDiffusion"): folder where the market data is stored
        :param asset_names (list or tuple, default=["DJ 50 EURO E", "S&P 500 NET EUR"]): list of asset names to be loaded
        :param allocation_frequency (str, default="monthly"): allocation frequency (available options: "monthly" and "daily"). The
        allocation frequency determines the time grid on which the RL agent changes the risky portfolio composition (alpha).
        :param target_volatility (float, default=5/100): target volatility level of the portfolio.
        :param tvs_spot_value (float, default=1.0): spot value of the target volatility strategy.
        :param option_strike (float, default=1.0): strike price of the option.
        :param option_maturity (float, default=2.0): maturity of the option expressed in yrf (it must be an integer number of years).
        :param neural_network_action_parameterization (str, default=None): neural network action parameterization (available: None,
        'baseline_strategy', 'black_strategy'). If None, the action of the RL agent is not parameterized (the NN output is the risky portfolio composition).
        If 'baseline_strategy', then the NN action is added to the baseline strategy to get the risky portfolio composition (the 'baseline_strategy' consists in
        solving the Black and Scholes problem path-wise). If 'black_strategy', then the NN action is added to the Black strategy to get the risky portfolio composition
        (the black strategy consists in the deterministic optimal solution found in the Black and Scholes environment).
        :param action_constraints (str, default=None): action constraints (available: None, 'long_only', 'long_short_limit'). If None, the risky allocation strategy is not constrained.
        If 'long_only', then the RL agent can only perform long positions normalized to 1. If 'long_only' is set and neural_network_action_parameterization is None then the action_space
        is hard coded to a Box([0, 1]). If 'long_short_limit', then the RL agent can perform long and short positions normalized to overall_long_position and overall_short_position respectively
        (overall_long_position and overall_short_position must be initialized).
        If 'long_short_limit' is selected, then no neural network action parameterization is allowed (not implemented yet).
        :param action_bound (float, default=5.0): action bound-> action space = Box([-np.fabs(action_bound), action_bound]). If action_constraints is set to 'long_short_limit',
        then the action space is set to a Box([0, 1]).
        :param overall_long_position (float, default=None): overall long position (if action_constraints is set to 'long_short_limit').
        :param overall_short_position (float, default=None): overall short position (if action_constraints is set to 'long_short_limit').
        :param n_sim_for_cache (int, default=int(2e3)): number of simulated paths cached in memory (the cache speed-up the computation because it exploits the vectorization provided by numpy).
        """

        # Store Pricing parameters for the option
        self.I_0 = tvs_spot_value
        self.I_t = tvs_spot_value
        self.strike_opt = option_strike
        self.T = option_maturity
        self.target_vol = target_volatility
        self.V_t = 0.0  # derivative value at time t
        self.V_t_plus_1 = 0  # derivative value at time t+1

        # Creation of the time grid describing the episode"""
        (
            self.observation_grid,
            self.state_index_grid,
            self.N_euler_grid,
        ) = build_allocation_time_grid(
            self.T, allocation_frequency, day_count_convention="ACT_365"
        )
        self.time_index = 0
        self.current_time = self.observation_grid[self.time_index]
        self.simulation_index = (
            0  # index of the simulation in the Monte Carlo simulation
        )
        self.Nsim = n_sim_for_cache  # number of genereated paths and cached in memory

        # Loading market curves and elements
        (
            discounting_curves,
            forward_curves,
            variance_curves,
            local_volatility_curves,
            correlation,
        ) = LoadFromTxt(asset_names, market_folder, local_vol_model=True)
        self.N_equity = len(asset_names)
        assert (
            len(forward_curves)
            == len(variance_curves)
            == len(local_volatility_curves)
            == len(correlation)
            == len(correlation.T)
            == self.N_equity
        )
        self.cholesky_matrix = np.linalg.cholesky(
            correlation
        )  # Cholesky decomposition of the correlation matrix
        self.spot_prices = np.zeros(self.N_equity)  # spot prices of the assets
        for i in range(self.N_equity):
            self.spot_prices[i] = forward_curves[i].spot
        self.intermediate_discounts = discounting_curves(self.T) / discounting_curves(
            self.observation_grid
        )
        self.mu_function = Drift(forward_curves)
        self.mu_time_grid = np.append(
            self.mu_function.T[self.mu_function.T < self.T], self.T
        )

        # Creation of the LV model
        self.model = LV_model(
            fixings=self.observation_grid[1:],
            forward_curve=forward_curves,
            local_vol_curve=local_volatility_curves,
            n_euler_grid=self.N_euler_grid,
            correlation_matrix=correlation,
            sampling="standard",
            return_grid_values_for_tvs=True,
        )  # do not insert the reference date in the simulating grid
        # Save forward values at the observation grid points (not the euler grid)
        self.forward_values_at_observation_grid = np.insert(
            self.model.forward_values, 0, self.spot_prices, axis=0
        )[
            self.state_index_grid, :
        ]  # shape (len(observation_grid), N_equity)
        # Collect discout factor at maturity date and the instantaneous interest rate on the euler grid for the TVS simulation
        euler_time_grid = self.model.euler_time_grid
        self.dt_vector = self.model.dt
        self.r_t_in_euler_grid = discounting_curves.r_t(
            np.append(0.0, euler_time_grid)
        )  # shape (N_euler_grid,) insert the referencedate and discard the last euler grid point
        self.discount_factor_at_maturity = discounting_curves(self.T)

        # Collect action space parameters
        allowed_action_constraints = ["long_only", "long_short_limit"]
        if action_constraints is None:  # free allocation strategy
            self.free_allocation_bounds = True
            self.long_only = False
            self.long_short_limit = False
            low_action = (
                np.ones(self.N_equity) * (-abs(np.fabs(action_bound))) - 1e-6
            )  # lower bound of the action space
            high_action = (
                np.ones(self.N_equity) * abs(np.fabs(action_bound)) + 1e-6
            )  # upper bound of the action space
        elif action_constraints == allowed_action_constraints[0]:  # long only strategy
            self.free_allocation_bounds = False
            self.long_only = True
            self.long_short_limit = False
            low_action = (
                np.ones(self.N_equity) * 1e-7
            )  # lower bound of the action space
            high_action = np.ones(self.N_equity)  # upper bound of the action space
        elif (
            action_constraints == allowed_action_constraints[1]
        ):  # overall position bounded strategy
            if overall_long_position is None or overall_short_position is None:
                raise ValueError(
                    f"Please specify the overall long and short positions for the {allowed_action_constraints[1]} constraint"
                )
            self.free_allocation_bounds = False
            self.long_only = False
            self.long_short_limit = True
            low_action = (
                np.ones(self.N_equity) * (-abs(np.fabs(action_bound))) - 1e-6
            )  # lower bound of the action space
            high_action = (
                np.ones(self.N_equity) * abs(np.fabs(action_bound)) + 1e-6
            )  # upper bound of the action space
            self.sum_long = overall_long_position
            self.sum_short = overall_short_position
        else:
            raise ValueError(
                f"Please specify an allowed action constraint: {allowed_action_constraints}"
            )

        # Collect neural network action parameterization
        allowed_action_parameterizations = ["baseline_strategy", "black_strategy"]
        if (
            neural_network_action_parameterization is None
        ):  # if no neural network action parameterization then the allocation strategy coincides with the NN output
            self.parameterized_action = False
            self.start_from_baseline = False
            self.start_from_black = False
        elif (
            neural_network_action_parameterization
            == allowed_action_parameterizations[0]
        ):
            self.parameterized_action = True
            self.start_from_baseline = True
            self.start_from_black = False
            self.mu_values_on_euler_grid = self.mu_function(
                np.append(0.0, euler_time_grid[:-1])
            )
            low_action = np.ones(self.N_equity) * (
                -np.fabs(action_bound)
            )  # lower bound of the action space
            high_action = np.ones(self.N_equity) * np.fabs(
                action_bound
            )  # upper bound of the action space
            if (
                action_constraints is not None
                and action_constraints != allowed_action_constraints[0]
            ):
                raise ValueError(
                    f"The {allowed_action_parameterizations[1]} is not implemented for the specified {action_constraints} action constraint"
                )
        elif (
            neural_network_action_parameterization
            == allowed_action_parameterizations[1]
        ):
            self.parameterized_action = True
            self.start_from_baseline = False
            self.start_from_black = True
            low_action = np.ones(self.N_equity) * (
                -np.fabs(action_bound)
            )  # lower bound of the action space
            high_action = np.ones(self.N_equity) * np.fabs(
                action_bound
            )  # upper bound of the action space
            nu_function = CholeskyTDependent(variance_curves, self.cholesky_matrix)
            alpha = Strategy()
            if action_constraints == allowed_action_constraints[0]:
                alpha.optimization_constrained(
                    mu=self.mu_function, nu=nu_function, n_trial=500, typo=1
                )
            elif action_constraints is None:
                alpha.Mark_strategy(mu=self.mu_function, nu=nu_function)
            else:
                raise ValueError(
                    f"The {allowed_action_parameterizations[1]} is not implemented for the specified {action_constraints} action constraint"
                )
            self.alpha_t = alpha(self.observation_grid[:-1])
        else:
            raise ValueError(
                f"Please specify an allowed neural network action parameterization: {allowed_action_parameterizations}"
            )

        # set the bounds of the action space
        self.action_space = spaces.Box(
            low=np.float32(low_action), high=np.float32(high_action)
        )
        # set the bounds of the observation space
        high = np.ones(self.N_equity) * 2.5
        low_bound = np.append(-high, 0.0)  # append the lower bound for the tvs level
        low_bound = np.append(low_bound, 0.0)  # append the lower bound for the time
        high_bound = np.append(
            high, self.I_0 * 10.0
        )  # append the upper bound for the tvs level
        high_bound = np.append(
            high_bound, self.T + 1.0 / 365
        )  # append the upper bound for the time
        assert len(low_bound) == len(high_bound)
        self.observation_space = spaces.Box(
            low=np.float32(low_bound), high=np.float32(high_bound)
        )
        # Useful elements for the simulation
        self.Identity = np.identity(self.N_equity)

    def step(self, action: np.ndarray) -> tuple:
        assert self.action_space.contains(action)

        if not self.parameterized_action:
            if self.free_allocation_bounds:
                risky_allocation_strategy = action
                overall_position = np.sum(risky_allocation_strategy)
            elif self.long_only:
                risky_allocation_strategy = action / np.sum(action)
                overall_position = 1.0
            elif self.long_short_limit:
                risky_allocation_strategy = sign_renormalization(
                    action, self.sum_long, self.sum_short
                )
                overall_position = np.sum(risky_allocation_strategy)
            else:
                raise ValueError("Please specify a valid allocation strategy")

        # update the current time and the equity stocks state
        self.time_index = self.time_index + 1
        self.current_time = self.observation_grid[self.time_index]
        self.current_logX = self.logX_t[self.time_index]
        dt = self.dt_vector[self.time_index - 1]
        index_euler_for_observation_date = (self.time_index - 1) * self.N_euler_grid
        # simulate the value of the tvs depending on the chosen risky_allocation_strategy
        for i in range(self.N_euler_grid):
            euler_grid_index = index_euler_for_observation_date + i
            sigma_matrix = self.instant_volatility[euler_grid_index] * self.Identity
            nu_matrix = sigma_matrix @ self.cholesky_matrix
            if i == 0:
                if self.parameterized_action:
                    if self.long_only:
                        if self.start_from_baseline:
                            parameterization = optimization_only_long(
                                self.mu_values_on_euler_grid[euler_grid_index],
                                nu_matrix,
                                n_trial=1,
                                guess=action,
                            )
                        elif self.start_from_black:
                            parameterization = self.alpha_t[self.time_index - 1]
                        risky_allocation_strategy = parameterization + action
                        risky_allocation_strategy[
                            risky_allocation_strategy < 0.0
                        ] = 0.0  # long only constraint
                        all_zeros_entries = np.all((risky_allocation_strategy == 0.0))
                        if all_zeros_entries:
                            risky_allocation_strategy = parameterization
                        else:
                            risky_allocation_strategy = (
                                risky_allocation_strategy
                                / np.sum(risky_allocation_strategy)
                            )
                        overall_position = 1.0
                    elif self.free_allocation_bounds:
                        if self.start_from_baseline:
                            parameterization = Markowitz_solution(
                                self.mu_values_on_euler_grid[euler_grid_index],
                                nu_matrix,
                                -1,
                            )
                        elif self.start_from_black:
                            parameterization = self.alpha_t[self.time_index - 1]
                        risky_allocation_strategy = parameterization + action
                        overall_position = np.sum(risky_allocation_strategy)
            product = risky_allocation_strategy @ nu_matrix
            norm = sqrt(product @ product)
            omega_coefficient = self.target_vol / norm
            self.I_t = self.I_t * (
                1.0
                + omega_coefficient
                * risky_allocation_strategy
                @ self.dS_S[euler_grid_index]
                + dt
                * self.r_t_in_euler_grid[euler_grid_index]
                * (1.0 - omega_coefficient * overall_position)
            )
        # Evaluation of the reward
        if self.free_allocation_bounds:
            black_optimal_allocation = Markowitz_solution(
                self.mu_function(self.current_time), nu_matrix, -1
            )
        elif self.long_only:
            black_optimal_allocation = optimization_only_long(
                self.mu_values_on_euler_grid[euler_grid_index], nu_matrix, n_trial=2
            )
        product = black_optimal_allocation @ nu_matrix
        norm = sqrt(product @ product)
        omega_coefficient = self.target_vol / norm
        mask = self.mu_time_grid > self.current_time
        grid = np.append(self.current_time, self.mu_time_grid[mask])
        mu_values = self.mu_function(grid[1:])
        local_drift = (
            np.sum(np.sum(black_optimal_allocation * mu_values, axis=1) * np.diff(grid))
            * omega_coefficient
        )
        forward_tvs = self.I_t * (
            np.exp(-local_drift) / self.intermediate_discounts[self.time_index]
        )
        if self.T - self.current_time > 1e-14:
            self.V_t_plus_1 = BS_European_option_closed_form(
                forward_tvs,
                self.strike_opt,
                self.T - self.current_time,
                self.intermediate_discounts[self.time_index],
                self.target_vol,
                1,
            )
        else:
            self.V_t_plus_1 = np.maximum(forward_tvs - self.strike_opt, 0.0)
        reward = self.V_t_plus_1 - self.V_t
        if self.current_time < self.T:
            done = False
            self.V_t = self.V_t_plus_1
        else:
            done = True
            self.simulation_index = self.simulation_index + 1
        state = np.append(
            self.current_logX, np.array([self.I_t / self.I_0, self.current_time])
        )
        return state, reward, done, {}

    def reset(self):
        if self.simulation_index == 0 or self.simulation_index == self.Nsim:
            self.simulations_logX = None  # free memory
            self.simulations_Vola = None  # free memory
            self.dS_S_simulations = None  # free memory
            S_t, self.simulations_Vola = self.model.simulate(
                random_generator=self.np_random, n_sim=self.Nsim
            )
            S_t = np.insert(S_t, 0, self.spot_prices, axis=1)
            self.dS_S_simulations = (S_t[:, 1:, :] - S_t[:, :-1, :]) / S_t[:, :-1, :]
            S_sliced = S_t[:, self.state_index_grid, :]
            self.simulations_logX = log(
                S_sliced / self.forward_values_at_observation_grid
            )
            self.simulation_index = 0

        self.current_time = 0.0
        self.time_index = 0
        self.I_t = self.I_0
        self.logX_t = self.simulations_logX[self.simulation_index]
        self.dS_S = self.dS_S_simulations[self.simulation_index]
        self.instant_volatility = self.simulations_Vola[self.simulation_index]
        state = np.append(
            self.logX_t[0], np.array([1.0, self.current_time])
        )  # initial state [stock price, tvs level, time]
        self.V_t = self.V_t_plus_1 = 0
        return state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        print()
        print("asset_history = ", self.asset_history)
        print("current time = ", self.current_time)

    def theoretical_price(self):
        return None  # there is no closed formulation for the theoretical price
