# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # <span style="color:green"> Monoasset test: Plain Vanilla Pricing with Time-Dependent Volatility </span>

# +
import sys

sys.path.insert(1, "./../../../src")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from pricing.closedforms import Price_to_BS_ImpliedVolatility
from pricing.montecarlo import MC_results
from pricing.pricing import (
    ACT_365,
    Black,
    DiscountingCurve,
    EquityForwardCurve,
    ForwardVariance,
    Vanilla_PayOff,
)
from pricing.read_market import LoadFromTxt

# +
idx = 0

folder = "../../market_data/Basket10Assets/"
asset_names_list = (
    "DJ 50 TR",
    "S&P 500 NET EUR",
    "MSCI EM MKT EUR",
    "I NKY NTR EUR",
    "DAX 30 E",
    "FTSE100 NTR E",
    "SMI TR EUR",
    "FTSEMIBN",
    "CAC 40 NTR",
    "HSI NTR EUR",
)
D, F, V, correlation_matrix = LoadFromTxt(
    asset_names_list, folder, "ATM_FWD", local_vol_model=False
)
F = F[idx]
V = V[idx]
assert V.asset_name == F.asset_name
# -

# ### Parameters of Simulation

t = 0
N_simulations = 6e5
T_max = 10
r = 1.0 / 100
sampling = "antithetic"
call_put = 1

# ### Simulation

maturities = np.loadtxt(folder + "maturities_vola_data_" + F.asset_name + ".txt")
BS = Black(maturities, forward_curve=F, variance_curve=V, sampling=sampling)
S_t = BS.simulate(n_sim=N_simulations, seed=14)

X_t = S_t / F(maturities)
strike = 1.0
pay = Vanilla_PayOff(X_t, strike, call_put, sampling=sampling)
result, err = MC_results(pay)
E = np.mean(X_t, axis=0)
T = maturities

# +
imp_volatility = np.zeros(len(T))
imp_volatility_plus = np.zeros(len(T))
imp_volatility_minus = np.zeros(len(T))
for i in range(len(T)):
    imp_volatility[i] = Price_to_BS_ImpliedVolatility(
        T[i], E[i], strike, result[i], call_put, 1.0
    )
    imp_volatility_minus[i] = Price_to_BS_ImpliedVolatility(
        T[i], E[i], strike, result[i] - err[i], call_put, 1.0
    )
    imp_volatility_plus[i] = Price_to_BS_ImpliedVolatility(
        T[i], E[i], strike, result[i] + err[i], call_put, 1.0
    )

y_lower = np.zeros(len(T))
y_upper = np.zeros(len(T))
for i in range(len(T)):
    if imp_volatility_minus[i] < imp_volatility_plus[i]:
        y_lower[i] = abs(imp_volatility[i] - imp_volatility_minus[i])
        y_upper[i] = abs(imp_volatility_plus[i] - imp_volatility[i])
    elif imp_volatility_minus[i] > imp_volatility_plus[i]:
        y_lower[i][j] = abs(imp_volatility[i][j] - imp_volatility_plus[i])
        y_upper[i][j] = abs(imp_volatility_minus[i] - imp_volatility[i])
# -

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
font = 13
plt.plot(T, V.spot_vol, "--")
plt.scatter(T, V.spot_vol, c="red")
plt.ylabel("$\sigma_{market}$", fontsize=font)
plt.xlabel("Maturity [yr]", fontsize=font)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
plt.title("Market Spot volatilities", fontsize=font)
plt.show()

# +
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
n_sigma = 1.0
plt.errorbar(
    T,
    abs(imp_volatility - V.spot_vol),
    yerr=[n_sigma * y_lower, n_sigma * y_upper],
    fmt="o",
    label="Monte Carlo",
)
plt.axhline(y=0, color="red", alpha=0.9, linestyle="-.", label="Target")
plt.xlabel("Maturity [yr]")
plt.ylabel("|IV - $\sigma_{market}$|")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
plt.grid(True)

plt.show()
