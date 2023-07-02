# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # <span style="color:purple"> Local Volatility Pricing Test </span>

# +
import sys

sys.path.insert(1, "../")

from time import time

import matplotlib.pyplot as plt
import numpy as np
from closedforms import Price_to_BS_ImpliedVolatility
from montecarlo import MC_Data_Blocking, MC_results
from read_market import LoadFromTxt

from pricing import (
    DiscountingCurve,
    EquityForwardCurve,
    ForwardVariance,
    LocalVolatilityCurve,
    LV_model,
    Vanilla_PayOff,
)

# +
idx = 0


folder = "../../market_data/Basket10Assets/"
asset_names_list = (
    "DJ 50 TR",
    "S&P 500 NET EUR",
    "MSCI EM MKT EUR",
    "I NKY NTR EUR",
    "FTSE100 NTR E",
    "SMI TR EUR",
    "DAX 30 E",
    "FTSEMIBN",
    "HSI NTR EUR",
)
D, F, V, LV, correlation_matrix = LoadFromTxt(
    asset_names_list, folder, "ATM_FWD", local_vol_model=True
)
F = F[idx]
V = V[idx]
LV = LV[idx]
assert V.asset_name == F.asset_name
assert LV.asset_name == F.asset_name

# +
spot_volatilities = V.market_volatility.T
vola_strikes = V.K
maturities = np.loadtxt(folder + "maturities_vola_data_" + F.asset_name + ".txt")
money_ness = np.zeros((len(vola_strikes), len(maturities)))
for j in range(len(vola_strikes)):
    for i in range(len(maturities)):
        money_ness[j, i] = vola_strikes[j] / F(maturities[i])

IV_curve = LocalVolatilityCurve(
    spot_volatilities, money_ness, maturities, V.asset_name, "piecewise"
)
# -

# %%time
N_grid = 160
sampling = "antithetic"
N_simulation = 1e5
expiries = LV.T
model = LV_model(
    fixings=expiries,
    local_vol_curve=LV,
    forward_curve=F,
    n_euler_grid=N_grid,
    sampling=sampling,
    return_grid_values_for_tvs=False,
)
X_t = model.simulate(n_sim=N_simulation, seed=21) / model.forward_values

# +
# %%time
kind = 1
expiries = LV.T
strikes = np.exp(LV.log_moneyness)
moneyness_matrix = np.exp(LV.log_moneyness)
n_strikes = len(moneyness_matrix)
n_expiries = len(expiries)
imp_volatility = np.zeros((n_strikes, n_expiries))
imp_volatility_plus = np.zeros((n_strikes, n_expiries))
imp_volatility_minus = np.zeros((n_strikes, n_expiries))
n_expiries = len(expiries)
E_X = np.mean(X_t, axis=0)
for j in range(n_expiries):
    for i in range(n_strikes):
        k = moneyness_matrix[i, j]
        if k >= 1:
            kind = 1  # 1 buono
        elif k < 1:
            kind = -1  # -1 buono
        option = Vanilla_PayOff(X_t, k, kind, sampling)
        result, err_result = MC_results(option)
        imp_volatility[i, j] = Price_to_BS_ImpliedVolatility(
            expiries[j], E_X[j], k, result[j], kind, 1.0
        )
        imp_volatility_plus[i, j] = Price_to_BS_ImpliedVolatility(
            expiries[j], E_X[j], k, result[j] + err_result[j], kind, 1.0
        )
        imp_volatility_minus[i, j] = Price_to_BS_ImpliedVolatility(
            expiries[j], E_X[j], k, result[j] - err_result[j], kind, 1.0
        )

y_lower = np.zeros(((n_strikes, n_expiries)))
y_upper = np.zeros(((n_strikes, n_expiries)))
for i in range(n_strikes):
    for j in range(n_expiries):
        if imp_volatility_minus[i][j] < imp_volatility_plus[i][j]:
            y_lower[i][j] = abs(imp_volatility[i][j] - imp_volatility_minus[i][j])
            y_upper[i][j] = abs(imp_volatility_plus[i][j] - imp_volatility[i][j])
        elif imp_volatility_minus[i][j] > imp_volatility_plus[i][j]:
            y_lower[i][j] = abs(imp_volatility[i][j] - imp_volatility_plus[i][j])
            y_upper[i][j] = abs(imp_volatility_minus[i][j] - imp_volatility[i][j])

# +
fig = plt.figure(1, figsize=(23, 22))
font = 13
plt.suptitle(LV.asset_name + " Local Volatility pricing", fontsize=font + 3)
n_sigma = 2.5
c = 3

for i in range(n_expiries):
    ax = fig.add_subplot(3, 4, i + 1)
    err_lower = n_sigma * y_lower.T[i]
    err_upper = n_sigma * y_upper.T[i]
    this_moneyness = moneyness_matrix[:, i]
    plt.plot(
        this_moneyness,
        imp_volatility[:, i] + n_sigma * err_upper,
        "r--",
        label="Monte Carlo",
    )
    plt.plot(this_moneyness, imp_volatility[:, i] - n_sigma * err_lower, "r--")
    plt.plot(
        this_moneyness, IV_curve(expiries[i], np.log(this_moneyness)), label="Market"
    )
    plt.xlabel("K/F", fontsize=font)
    plt.ylabel(r"IV", fontsize=font)
    plt.xticks(fontsize=font - 1)
    plt.yticks(fontsize=font - 1)
    plt.title("Maturity = " + str(round(expiries[i], c)) + " [yr]", fontsize=font)
    plt.legend()
