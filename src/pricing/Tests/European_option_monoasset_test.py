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

# # <span style="color:blue"> Monoasset test: Pricing European Option </span>

# +
import sys

sys.path.insert(1, "../")

import time

import matplotlib.pyplot as plt
import numpy as np
from closedforms import BS_European_option_closed_form, Price_to_BS_ImpliedVolatility
from matplotlib import ticker
from montecarlo import MC_Data_Blocking, MC_results
from numpy import exp, log

from pricing import (
    Black,
    DiscountingCurve,
    EquityForwardCurve,
    ForwardVariance,
    Vanilla_PayOff,
)

# -

# ### Parameters of Simulation

r = 1 / 100
t = 0
volatility = 20 / 100
spot_price = 150.0
T_max = 10.0
N_simulations = 5e5
N_block = 100
reference_date = 0
sampling = "antithetic"

# ### Market Data

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
    reference_date,
    spot_price,
    D,
    repo_dates=np.array([0.0, T_max]),
    repo_rates=np.array([0.1 / 100, 0.1 / 100]),
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
"""Time dependent volatility"""
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

#
# ### Simulation

# +
# %%time
kind = 1
maturities = np.arange(1, 31) * 0.15
B_model = Black(maturities, forward_curve=F, variance_curve=V, sampling=sampling)
S_t = B_model.simulate(n_sim=N_simulations, seed=12)
X_t = S_t / F(maturities)
E_X = np.mean(X_t, axis=0)
K_norm = 1
pay_normalized = Vanilla_PayOff(X_t, K_norm, kind, sampling)  # ATM forward pricing

"""Calculating closed form"""
call_black = BS_European_option_closed_form(
    F(maturities), F(maturities), maturities, D(maturities), volatility, kind
)
# -

# ### Convergence Check

# #### In term of price

n = 29
x, result, result_err = MC_Data_Blocking(pay_normalized[:, n], N_block)
mean_price = result * D(maturities[n]) * F(maturities[n])
err_price = result_err * D(maturities[n]) * F(maturities[n])

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.errorbar(x, mean_price, yerr=err_price, label="Monte Carlo")
plt.axhline(y=call_black[n], color="red", linestyle="-", label="B&S")
plt.xlabel("# MC throws")
plt.title(
    "log(K/F) = " + str(log(K_norm)) + " - Maturity = " + str(round(maturities[n], 1))
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

# +
z = np.std(pay_normalized.T[n]) * F(maturities[n]) * D(maturities[n])
y = np.sqrt(x)
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.plot(x, err_price, label="Monte Carlo error")
plt.plot(x, z / y, label="Expected error")
plt.xlabel("#throws")
plt.title(
    "log(K/F) = " + str(log(K_norm)) + " - Maturity = " + str(round(maturities[n], 1))
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
# -

# #### In term of implied volatility

# +
imp_volatility_mean = np.zeros(N_block)
imp_volatility_plus = np.zeros(N_block)
imp_volatility_minus = np.zeros(N_block)
for i in range(N_block):
    imp_volatility_mean[i] = Price_to_BS_ImpliedVolatility(
        maturities[n], E_X[n], 1.0, result[i], kind, 1.0
    )
    imp_volatility_plus[i] = Price_to_BS_ImpliedVolatility(
        maturities[n], E_X[n], 1.0, result[i] + result_err[i], kind, 1.0
    )
    imp_volatility_minus[i] = Price_to_BS_ImpliedVolatility(
        maturities[n], E_X[n], 1.0, result[i] - result_err[i], kind, 1.0
    )

y_lower = np.zeros(N_block)
y_upper = np.zeros(N_block)
for i in range(N_block):
    if imp_volatility_minus[i] < imp_volatility_plus[i]:
        y_lower[i] = abs(imp_volatility_mean[i] - imp_volatility_minus[i])
        y_upper[i] = abs(imp_volatility_plus[i] - imp_volatility_mean[i])
    elif imp_volatility_minus[i] > imp_volatility_plus[i]:
        y_lower[i] = abs(imp_volatility[i] - imp_volatility_plus[i])
        y_upper[i] = abs(imp_volatility_minus[i] - imp_volatility_mean[i])

# -

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.title(
    r"log$(K/F)$ = "
    + str(log(K_norm))
    + " -- Maturity = "
    + str(round(maturities[n], 1))
    + " [yr]",
    fontsize=13,
)
plt.errorbar(x, imp_volatility_mean, yerr=[y_lower, y_upper], label="Monte Carlo")
plt.axhline(y=volatility, lw=2, c="xkcd:red", linestyle="-.", label="BS model")
plt.xlabel("# MC throws", fontsize=13)
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.ylabel("IV", fontsize=13)
plt.legend()
plt.show()

# ### Implied volatility in function of log Moneyness log(K/F)

# +
# %%time
kind = -1
logmoneyness = np.arange(-10, 11) * 0.1
imp_volatility = np.zeros((len(logmoneyness), len(maturities)))
imp_volatility_plus = np.zeros((len(logmoneyness), len(maturities)))
imp_volatility_minus = np.zeros((len(logmoneyness), len(maturities)))
for i in range(len(logmoneyness)):
    if logmoneyness[i] >= 0.0:
        kind = 1
    elif logmoneyness[i] < 0.0:
        kind = -1
    K = np.exp(logmoneyness[i])
    option = Vanilla_PayOff(X_t, K, kind, sampling=sampling)
    result, err_result = MC_results(option)
    for j in range(len(maturities)):
        imp_volatility[i][j] = Price_to_BS_ImpliedVolatility(
            maturities[j], E_X[j], K, result[j], kind, 1.0
        )
        imp_volatility_plus[i][j] = Price_to_BS_ImpliedVolatility(
            maturities[j], E_X[j], K, result[j] + err_result[j], kind, 1.0
        )
        imp_volatility_minus[i][j] = Price_to_BS_ImpliedVolatility(
            maturities[j], E_X[j], K, result[j] - err_result[j], kind, 1.0
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
plt.suptitle("European Option: Implied volatility at fixed log-moneyness", fontsize=18)
n_sigma = 2.55  # how many sigma for the errorbars
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
    y=volatility, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("Maturity", fontsize=13)
plt.ylabel("IV", fontsize=13)
# plt.ylim(0.18,0.235)
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
    y=volatility, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
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
    y=volatility, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("Maturity", fontsize=13)
plt.ylabel("IV", fontsize=13)
# plt.ylim(0.05,0.25)
plt.legend()
plt.show()

# +
plt.figure(figsize=(22, 13))
plt.suptitle("European Option: Implied volatility at fixed maturity", fontsize=18)
n_sigma = 2.55  # how many sigma for the errorbars

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
    y=volatility, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("log(K/F)", fontsize=13)
plt.ylabel("IV", fontsize=13)
plt.title("Maturity = " + str(round(maturities[num[i]], c)))
# plt.ylim(0.193,0.205)
plt.legend()

num = [10]
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
    y=volatility, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("log(K/F)", fontsize=13)
plt.ylabel("IV", fontsize=13)
plt.title("Maturity: " + str(round(maturities[num[i]], c)))
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
    y=volatility, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("log(K/F)", fontsize=13)
plt.ylabel("IV", fontsize=13)
plt.title("Maturity = " + str(round(maturities[num[i]], c)))
plt.legend()
plt.show()
