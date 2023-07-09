import sys
from typing import List, Tuple

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

sys.path.insert(1, "./src")
from pricing.closedforms import BS_European_option_closed_form
from pricing.pricing import Black
from pricing.read_market import LoadFromTxt
from pricing.targetvol import (
    CholeskyTDependent,
    Drift,
    Strategy,
    TargetVolatilityStrategy,
    TVSForwardCurve,
)
from reinforcement.envs.utils import build_allocation_time_grid, sign_renormalization


class TVS_BS_ENV(gym.Env):
    """Target volatility strategy Option environment for the Black-Scholes model."""

    def __init__(
        self,
        market_folder: str = "./src/market_data/FakeSmilesDisplacedDiffusion",
        asset_names: List[str] = ["DJ 50 EURO E", "S&P 500 NET EUR"],
        allocation_frequency: str = "monthly",
        target_volatility: float = 5 / 100,
        tvs_spot_value: float = 1.0,
        option_strike: float = 1.0,
        option_maturity: float = 2.0,
        action_constraints: str = None,
        action_bound: float = 5.0,
        overall_long_position: float = None,
        overall_short_position: float = None,
        n_sim_for_cache: int = int(2e3),
    ) -> None:
        """
        Target volatility strategy Option RL environment with a Black and Scholes model for the assets. In this environment the action space and the observation
        space are continuous. The observation state is a vector of size (n_assets + 2) with the following structure: [stock price, tvs level, time]
        and the state space is the following block: lower_bound = [-2.5, 0., 0.]; upper_bound = [2.5, 10 * tvs_spot_value, maturity].
        :param market_folder (str, default="../market_data/FakeSmilesDisplacedDiffusion"): folder where the market data is stored
        :param asset_names (list or tuple, default=["DJ 50 EURO E", "S&P 500 NET EUR"]): list of asset names to be loaded
        :param allocation_frequency (str, default="monthly"): allocation frequency (available options: "monthly" and "daily"). The
        allocation frequency determines the time grid on which the RL agent changes the risky portfolio composition (alpha).
        :param target_volatility (float, default=5/100): target volatility level of the portfolio.
        :param tvs_spot_value (float, default=1.0): spot value of the target volatility strategy.
        :param option_strike (float, default=1.0): strike price of the option.
        :param option_maturity (float, default=2.0): maturity of the option expressed in yrf (it must be an integer number of years).
        :param action_constraints (str, default=None): action constraints (available: None, 'long_only', 'long_short_limit'). If None, the risky allocation strategy is not constrained.
        If 'long_only', then the RL agent can only perform long positions normalized to 1. If 'long_only' is set then the action_space is hard coded to a Box([0, 1]).
        If 'long_short_limit', then the RL agent can perform long and short positions normalized to overall_long_position and overall_short_position respectively
        (overall_long_position and overall_short_position must be initialized).
        :param action_bound (float, default=5.0): action bound-> action space = Box([-np.fabs(action_bound), action_bound]). If action_constraints is set to 'long_short_limit',
        then the action space is set to a Box([0, 1]).
        :param overall_long_position (float, default=None): overall long position (if action_constraints is set to 'long_short_limit').
        :param overall_short_position (float, default=None): overall short position (if action_constraints is set to 'long_short_limit').
        :param n_sim_for_cache (int, default=int(2e3)): number of simulated paths cached in memory (the cache speed-up the computation because it exploits the vectorization provided by numpy).
        """

        self.I_0 = tvs_spot_value
        self.strike_opt = option_strike
        self.T = option_maturity
        self.target_vol = target_volatility

        # Creation of the time grid describing the episode"""
        self.observation_grid, _, _ = build_allocation_time_grid(
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
            self.discounting_curve,
            forward_curves,
            variance_curves,
            correlation,
        ) = LoadFromTxt(asset_names, market_folder, local_vol_model=False)
        self.N_equity = len(asset_names)
        assert len(forward_curves) == len(variance_curves) == self.N_equity
        correlation = np.array(([1.0, 0.0], [0.0, 1.0]))
        self.cholesky_matrix = np.linalg.cholesky(
            correlation
        )  # Cholesky decomposition of the correlation matrix
        self.spot_prices = np.zeros(self.N_equity)  # spot prices of the assets
        self.discount_at_maturity = self.discounting_curve(self.T)
        for i in range(self.N_equity):
            self.spot_prices[i] = forward_curves[i].spot

        # Creation of the BS model
        self.model = Black(
            fixings=self.observation_grid[1:],
            forward_curve=forward_curves,
            variance_curve=variance_curves,
            correlation_matrix=correlation,
            sampling="standard",
        )
        # Set target volatility strategy classes
        self.mu_function = Drift(forward_curves)
        self.nu_function = CholeskyTDependent(
            variance_curves, np.linalg.cholesky(correlation)
        )
        self.TVSF = TVSForwardCurve(
            reference=0,
            vola_target=self.target_vol,
            spot_price=self.I_0,
            mu=self.mu_function,
            nu=self.nu_function,
            discounting_curve=self.discounting_curve,
            strategy=None,
        )

        # Collect action space parameters
        allowed_action_constraints = ["long_only", "long_short_limit"]
        if action_constraints is None:  # free allocation strategy
            self.free_allocation_bounds = True
            self.long_only = False
            self.long_short_limit = False
            low_action = (
                np.ones(self.N_equity) * (-abs(action_bound)) - 1e-6
            )  # lower bound of the action space
            high_action = (
                np.ones(self.N_equity) * abs(action_bound) + 1e-6
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
                    f"Please specify the overall long and short positions for the "
                    f"{allowed_action_constraints[1]} constraint"
                )
            self.free_allocation_bounds = False
            self.long_only = False
            self.long_short_limit = True
            low_action = (
                np.ones(self.N_equity) * (-abs(action_bound)) - 1e-6
            )  # lower bound of the action space
            high_action = (
                np.ones(self.N_equity) * abs(action_bound) + 1e-6
            )  # upper bound of the action space
            self.sum_long = overall_long_position
            self.sum_short = overall_short_position
        else:
            raise ValueError(
                f"Please specify an allowed action constraint: {allowed_action_constraints}"
            )
        # set the bounds of the action space
        self.action_space = spaces.Box(
            low=np.float32(low_action), high=np.float32(high_action)
        )

        # Set state space
        high = np.ones(self.N_equity) * 2.5
        low_bound = np.append(-high, 0.0)
        high_bound = np.append(high, self.T + 1.0 / 365)
        self.observation_space = spaces.Box(
            low=np.float32(low_bound), high=np.float32(high_bound)
        )

    def step(self, action: np.ndarray) -> Tuple:
        assert self.action_space.contains(action)

        if self.free_allocation_bounds:
            risky_allocation_strategy = action
        elif self.long_only:
            risky_allocation_strategy = action / np.sum(action)
        elif self.long_short_limit:
            risky_allocation_strategy = sign_renormalization(
                action, self.sum_long, self.sum_short
            )
        else:
            raise ValueError("Please specify a valid allocation strategy")

        if self.time_index == 0:
            self.alpha_t = risky_allocation_strategy
        else:
            self.alpha_t = np.vstack([self.alpha_t, risky_allocation_strategy])

        self.time_index = self.time_index + 1
        self.current_time = self.observation_grid[self.time_index]
        self.current_asset = self.S_t[self.time_index]

        if self.current_time < self.T:
            done = False
            reward = 0.0
        else:
            done = True
            alpha = Strategy(strategy=self.alpha_t, dates=self.observation_grid[:-1])
            self.TVSF.set_strategy(alpha)
            TVS = TargetVolatilityStrategy(
                fixings=np.array([self.T]), forward_curve=self.TVSF, sampling="standard"
            )
            I_t = TVS.simulate(random_generator=self.np_random)[0, 0]
            reward = np.maximum(I_t - self.strike_opt, 0.0) * self.discount_at_maturity
            self.simulation_index = self.simulation_index + 1

        state = np.append(self.current_asset, self.current_time)
        return state, reward, done, {}

    def reset(self) -> np.ndarray:
        if self.simulation_index == 0 or self.simulation_index == self.Nsim:
            self.simulations = None  # clear memory
            self.simulations = self.model.simulate(
                random_generator=self.np_random,
                n_sim=self.Nsim,
                return_log_martingale=True,
            )
            self.simulations = np.insert(
                self.simulations, 0, np.zeros(self.N_equity), axis=1
            )  # add initial asset price
            self.simulation_index = 0
        self.current_time = 0.0
        self.time_index = 0
        self.S_t = self.simulations[self.simulation_index]
        self.alpha_t = np.array([])
        state = np.append(self.S_t[self.time_index], self.current_time)
        return state

    def seed(self, seed=None) -> List:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human") -> None:
        pass

    def theoretical_price(self) -> float:
        optimal_strategy = Strategy()
        optimal_strategy.Mark_strategy(mu=self.mu, nu=self.nu)
        TVSF = TVSForwardCurve(
            reference=0.0,
            vola_target=self.target_vol,
            spot_price=self.I_0,
            mu=self.mu,
            nu=self.nu,
            discounting_curve=self.discounting_curve,
            strategy=optimal_strategy,
        )
        return BS_European_option_closed_form(
            TVSF(self.T),
            self.strike_option,
            self.T,
            self.discounting_curve(self.T),
            self.target_vol,
            1.0,
        )
