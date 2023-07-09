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

# # <span style="color:orange"> Monoasset test: Pricing Geometric Average Asian Option </span>

# +
import sys

sys.path.insert(1, "./../../../src")


import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from numpy import exp, log
from scipy.stats.mstats import gmean

from pricing.closedforms import (
    BS_European_option_closed_form,
    GA_Asian_option_closed_form,
    Price_to_BS_ImpliedVolatility,
    interest_rate_asian,
    volatility_asian,
)
from pricing.montecarlo import MC_Data_Blocking, MC_results
from pricing.pricing import (
    Black,
    DiscountingCurve,
    EquityForwardCurve,
    ForwardVariance,
    Vanilla_PayOff,
)

# -

# ### Parameters of Simulation

t = 0
spot_price = 150.0
N_simulations = 5e5
N_block = 100
N_averages = 30
T_max = 10
maturity = 4.0
dates = np.linspace(0.0, maturity, N_averages)
r = 1.0 / 100
volatility = 50.0 / 100
asian_vola = volatility_asian(N_averages, volatility)
sampling = "antithetic"

# ### Market Data
#

# +
"""Discounut Factors"""
zero_interest_rate = np.array([r, r, r])
zero_interest_rate_dates = np.array([0.0, 5, T_max])
d = exp(-zero_interest_rate * zero_interest_rate_dates)  # market discount factors
D = DiscountingCurve(
    reference=t, discounts=d, discount_dates=zero_interest_rate_dates
)  # discounting curve

plt.figure(figsize=(15, 4))

plt.subplot(1, 2, 1)
plt.step(zero_interest_rate_dates, D.R(zero_interest_rate_dates), color="red")
plt.xlabel("time")
plt.ylabel("zero interest rate")

plt.subplot(1, 2, 2)
x = np.linspace(0.0, 6, 100)
plt.plot(x, D(x))
plt.xlabel("time")
plt.ylabel("Discount curve")
plt.show()

# +
"""Forward curve and repo rates"""
F = EquityForwardCurve(
    reference=t,
    discounting_curve=D,
    spot=spot_price,
    repo_dates=np.array([0.0, T_max]),
    repo_rates=np.array([0.0, 0.0]),
)

plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.step(F.T, F.q_values, color="red")
plt.xlabel("time")
plt.ylabel("repo rate")

plt.subplot(1, 2, 2)
plt.plot(x, F(x))
plt.xlabel("time")
plt.ylabel("Forward curve")
plt.show()

# +
"""Time dependent Volatility"""
K_spot_vola = np.array([spot_price, 200])
spot_vol = np.array(([volatility, volatility], [volatility, volatility]))
spot_vol_dates = np.array([0.1, T_max])
V = ForwardVariance(
    reference=t,
    maturity_dates=spot_vol_dates,
    strikes=K_spot_vola,
    market_volatility_matrix=spot_vol,
    strike_interp=spot_price,
)

plt.figure(figsize=(7, 4))
time = V.T.tolist()
vola = V.forward_vol.tolist()
time.insert(0, V.T[0])
time.insert(0, 0)
vola.insert(0, V.forward_vol[0])
vola.insert(0, V.forward_vol[0])
plt.step(time, vola)
plt.xlabel("Time")
plt.ylabel("forward volatility")
plt.show()
# -

# ### Simulation

# +
# %%time
B_model = Black(dates, forward_curve=F, variance_curve=V, sampling=sampling)
kind = 1
S_t = B_model.simulate(n_sim=N_simulations, seed=8)
G_mean = gmean(S_t, axis=1)
forward_g_asian = spot_price * exp(
    interest_rate_asian(N_averages, r, volatility) * maturity
)
X_t = G_mean / forward_g_asian
E_X = np.mean(X_t)
K_norm = 1
pay_normalized = Vanilla_PayOff(
    X_t, K_norm, call_put=kind, sampling=sampling
)  # ATM forward pricing

"""Calculating closed form"""
right = GA_Asian_option_closed_form(
    F(maturity), forward_g_asian, maturity, D(maturity), volatility, N_averages, kind
)
# -

# ### Convergence Check

x, result, result_err = MC_Data_Blocking(pay_normalized, N_block)
mean_price = result * D(maturity) * forward_g_asian
err_price = result_err * D(maturity) * forward_g_asian

# #### In term of price

nu = dates[1] - dates[0]
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.axhline(y=right, color="red", linestyle="-", label="B&S")
plt.errorbar(x, mean_price, yerr=err_price, label="Monte Carlo")
plt.xlabel("#throws")
plt.title(
    "log(K/F) = "
    + str(log(K_norm))
    + " - Maturity = "
    + str(maturity)
    + " - Time between averaging points = "
    + str(round(nu, 2))
)
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.legend()
if kind == 1:
    plt.ylabel("Call option price")
elif kind == -1:
    plt.ylabel("Put option price")
plt.show()

z = np.std(pay_normalized) * D(maturity) * forward_g_asian
y = np.sqrt(x)
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.plot(x, err_price, label="Monte Carlo error")
plt.plot(x, z / y, label="Expected error")
plt.xlabel("#throws")
plt.title(
    "log(K/F) = "
    + str(log(K_norm))
    + " - Maturity = "
    + str(maturity)
    + " - Time between averaging points = "
    + str(round(nu, 2))
)
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.legend()
if kind == 1:
    plt.ylabel("Error call option price")
elif kind == -1:
    plt.ylabel("Error put option price")
plt.show()

# #### In term of implied volatility

imp_volatility_mean = np.zeros(N_block)
imp_volatility_plus = np.zeros(N_block)
imp_volatility_minus = np.zeros(N_block)
for i in range(N_block):
    imp_volatility_mean[i] = Price_to_BS_ImpliedVolatility(
        maturity, E_X, 1.0, result[i], kind, 1.0
    )
    imp_volatility_plus[i] = Price_to_BS_ImpliedVolatility(
        maturity, E_X, 1.0, result[i] + result_err[i], kind, 1.0
    )
    imp_volatility_minus[i] = Price_to_BS_ImpliedVolatility(
        maturity, E_X, 1.0, result[i] - result_err[i], kind, 1.0
    )
y_lower = np.zeros(N_block)
y_upper = np.zeros(N_block)
for i in range(N_block):
    if imp_volatility_minus[i] < imp_volatility_plus[i]:
        y_lower[i] = abs(imp_volatility_mean[i] - imp_volatility_minus[i])
        y_upper[i] = abs(imp_volatility_plus[i] - imp_volatility_mean[i])
    elif imp_volatility_minus[i] > imp_volatility_plus[i]:
        y_lower[i] = abs(imp_volatility_mean[i] - imp_volatility_plus[i])
        y_upper[i] = abs(imp_volatility_minus[i] - imp_volatility_mean[i])

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.title(
    r"log$(K/F)$ = "
    + str(log(K_norm))
    + " -- Maturity = "
    + str(round(maturity, 1))
    + " [yr] -- m = 30"
)
plt.errorbar(x, imp_volatility_mean, yerr=[y_lower, y_upper], label="Monte Carlo")
plt.axhline(y=asian_vola, lw=2, c="xkcd:red", linestyle="-", label="BS model")
plt.xlabel("# MC throws")
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.ylabel("IV")
plt.legend()

# ### Implied volatility in function of Log-moneyness log(K/F)

# +
# %%time
logmoneyness = np.arange(-10, 11) * 0.1
maturities = np.arange(1, 31) * 0.15
N_averages = 30
imp_volatility = np.zeros((len(logmoneyness), len(maturities)))
imp_volatility_plus = np.zeros((len(logmoneyness), len(maturities)))
imp_volatility_minus = np.zeros((len(logmoneyness), len(maturities)))
random_generator = np.random
random_generator.seed(8)
for i in range(len(maturities)):
    dates = np.linspace(0.0, maturities[i], N_averages)
    B_model = Black(dates, forward_curve=F, variance_curve=V, sampling=sampling)
    S_t = B_model.simulate(n_sim=N_simulations, random_generator=random_generator)
    G_mean = gmean(S_t, axis=1)
    for j in range(len(logmoneyness)):
        if logmoneyness[j] >= 0.0:
            kind = 1
        elif logmoneyness[j] < 0.0:
            kind = -1
        K = np.exp(logmoneyness[j])
        forward = spot_price * exp(
            interest_rate_asian(N_averages, r, volatility) * maturities[i]
        )
        X_t = G_mean / forward
        E = np.mean(X_t, axis=0)
        option = Vanilla_PayOff(X_t, K, kind, sampling)
        result, err_result = MC_results(option)
        imp_volatility[j][i] = Price_to_BS_ImpliedVolatility(
            maturities[i], E, K, result, kind, 1.0
        )
        imp_volatility_plus[j][i] = Price_to_BS_ImpliedVolatility(
            maturities[i], E, K, result + err_result, kind, 1.0
        )
        imp_volatility_minus[j][i] = Price_to_BS_ImpliedVolatility(
            maturities[i], E, K, result - err_result, kind, 1.0
        )

y_lower = np.zeros((len(logmoneyness), len(maturities)))
y_upper = np.zeros((len(logmoneyness), len(maturities)))
for i in range(len(logmoneyness)):
    for j in range(len(maturities)):
        if imp_volatility_minus[i][j] < imp_volatility_plus[i][j]:
            y_lower[i][j] = abs(imp_volatility[i][j] - imp_volatility_minus[i][j])
            y_upper[i][j] = abs(imp_volatility_plus[i][j] - imp_volatility[i][j])
        elif imp_volatility_minus[i][j] > imp_volatility_plus[i][j]:
            y_lower[i][j] = abs(imp_volatility[i][j] - imp_volatility_plus[i][j])
            y_upper[i][j] = abs(imp_volatility_minus[i][j] - imp_volatility[i][j])

# +
plt.figure(figsize=(22, 13))
plt.suptitle(
    "Geometric Average Asian Option: Implied volatility at fixed log-moneyness",
    fontsize=18,
)
n_sigma = 2  # how many sigma for the errorbars
i = 0
num = [0]
c = 2
plt.subplot(2, 3, 1)
err_lower = n_sigma * y_lower[num[i]]
err_upper = n_sigma * y_upper[num[i]]
plt.errorbar(
    maturities,
    imp_volatility[num[i]],
    yerr=[err_lower, err_upper],
    fmt="o",
    color="green",
    label="Monte Carlo",
)
plt.title("log(K/F) = " + str(round(logmoneyness[num[i]], c)), fontsize=13)
plt.axhline(
    y=asian_vola, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("Maturity", fontsize=13)
plt.ylabel("IV", fontsize=13)
# plt.ylim(0.038,0.12)
plt.legend()

num = [10]
plt.subplot(2, 3, 2)
err_lower = n_sigma * y_lower[num[i]]
err_upper = n_sigma * y_upper[num[i]]
plt.errorbar(
    maturities,
    imp_volatility[num[i]],
    yerr=[err_lower, err_upper],
    fmt="o",
    color="red",
    label="Monte Carlo",
)
plt.title("log(K/F) = " + str(round(logmoneyness[num[i]], c)), fontsize=13)
plt.axhline(
    y=asian_vola, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("Maturity", fontsize=13)
plt.ylabel("IV", fontsize=13)
plt.legend()

num = [20]
plt.subplot(2, 3, 3)
err_lower = n_sigma * y_lower[num[i]]
err_upper = n_sigma * y_upper[num[i]]
plt.errorbar(
    maturities,
    imp_volatility[num[i]],
    yerr=[err_lower, err_upper],
    fmt="o",
    color="blue",
    label="Monte Carlo",
)
plt.title("log(K/F) = " + str(round(logmoneyness[num[i]], c)), fontsize=13)
plt.axhline(
    y=asian_vola, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("Maturity", fontsize=13)
plt.ylabel("IV", fontsize=13)
# plt.ylim(0.04,0.12)
plt.legend()
plt.show()

# +
plt.figure(figsize=(22, 13))
plt.suptitle(
    "Geometric Average Asian Option: Implied volatility at fixed maturity", fontsize=18
)
n_sigma = 2  # how many sigma for the errorbars

num = [0]
c = 2
plt.subplot(2, 3, 1)
err_lower = n_sigma * y_lower.T[num[i]]
err_upper = n_sigma * y_upper.T[num[i]]
plt.errorbar(
    logmoneyness,
    imp_volatility.T[num[i]],
    yerr=[err_lower, err_upper],
    fmt="o",
    color="green",
    label="Monte Carlo",
)
plt.axhline(
    y=asian_vola, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("log(K/F)", fontsize=13)
plt.ylabel("IV", fontsize=13)
plt.title("Maturity = " + str(round(maturities[num[i]], c)))
# plt.ylim(0.10,0.12)
plt.legend()

num = [11]
plt.subplot(2, 3, 2)
err_lower = n_sigma * y_lower.T[num[i]]
err_upper = n_sigma * y_upper.T[num[i]]
plt.errorbar(
    logmoneyness,
    imp_volatility.T[num[i]],
    yerr=[err_lower, err_upper],
    fmt="o",
    color="red",
    label="Monte Carlo",
)
plt.axhline(
    y=asian_vola, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("log(K/F)", fontsize=13)
plt.ylabel("IV", fontsize=13)
plt.title("Maturity = " + str(round(maturities[num[i]], c)))
# plt.ylim(0.039,0.12)
plt.legend()

num = [29]
plt.subplot(2, 3, 3)
err_lower = n_sigma * y_lower.T[num[i]]
err_upper = n_sigma * y_upper.T[num[i]]
plt.errorbar(
    logmoneyness,
    imp_volatility.T[num[i]],
    yerr=[err_lower, err_upper],
    fmt="o",
    color="blue",
    label="Monte Carlo",
)
plt.axhline(
    y=asian_vola, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("log(K/F)", fontsize=13)
plt.ylabel("IV", fontsize=13)
plt.title("Maturity = " + str(round(maturities[num[i]], c)))
plt.legend()
plt.show()
