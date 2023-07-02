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

# # <span style="color:red"> Multiasset test: Pricing Geometric Average Basket Option </span>

# +
import sys

sys.path.insert(1, "../")

import time

import matplotlib.pyplot as plt
import numpy as np
from closedforms import (
    GAM_Basket_option_closed_form,
    Price_to_BS_ImpliedVolatility,
    forward_basket,
    volatility_basket,
)
from matplotlib import ticker
from montecarlo import MC_Data_Blocking, MC_results
from numpy import exp, log
from scipy.stats.mstats import gmean

from pricing import (
    Black,
    DiscountingCurve,
    EquityForwardCurve,
    ForwardVariance,
    Vanilla_PayOff,
)

# -

# ### Parameters of the Simulation

r = 1.0 / 100
t = 0
spot_price = np.array([110.0, 120.0, 97.0, 133.0])
T_max = 10
N_simulation = 1e6
N_block = 100
volatility = np.array([20.0, 30.0, 25.0, 32.0]) / 100
corr = np.array(
    (
        [1, 0.15, 0.10, 0.20],
        [0.15, 1.0, -0.05, 0.18],
        [0.1, -0.05, 1.0, 0.13],
        [0.20, 0.18, 0.13, 1],
    )
)
sampling = "antithetic"
N_equity = len(corr)
sigma_basket = volatility_basket(volatility, corr)

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
F = []  # list of all equity forward curves
repos = np.array([0.5, 0.9, 0.38, 0.44])
for i in range(N_equity):
    F.append(
        EquityForwardCurve(
            reference=t,
            spot=spot_price[i],
            discounting_curve=D,
            repo_rates=np.array([repos[i], repos[i]]) / 100,
            repo_dates=np.array([0.0, T_max]),
        )
    )

x = np.linspace(0, 5, 100)
for i in range(len(F)):
    plt.plot(x, F[i](x), label="Equity " + str(i + 1))
plt.legend()
plt.xlabel("time")
plt.ylabel("Forward")
plt.show()

# +
T1 = np.array([0.08, 0.17, 0.25, 0.33, 0.42, 0.50, 1.0, 2.0, T_max])
K1 = np.array([spot_price[0], 164, 172, 180])

sigma1 = np.array(
    (
        [
            volatility[0],
            volatility[0],
            volatility[0],
            volatility[0],
            volatility[0],
            volatility[0],
            volatility[0],
            volatility[0],
            volatility[0],
        ],
        [32, 29.1, 28.9, 29.3, 29.4, 29.4, 29.6, 30.6, 30.63],
        [33.6, 29.3, 29, 29.3, 29.3, 29.3, 29.3, 30.6, 30.59],
        [35, 29.7, 29.4, 29.5, 29.4, 29.3, 29.3, 30.5, 30.46],
    )
)
sigma1 = sigma1

T2 = np.array([0.08, 0.17, 0.25, 0.33, 0.42, 0.50, 1.0, 2.0, T_max])
K2 = np.array([spot_price[1], 164, 172, 180])
sigma2 = np.array(
    (
        [
            volatility[1],
            volatility[1],
            volatility[1],
            volatility[1],
            volatility[1],
            volatility[1],
            volatility[1],
            volatility[1],
            volatility[1],
        ],
        [32, 29.1, 28.9, 29.3, 29.4, 29.4, 29.6, 30.6, 30.63],
        [33.6, 29.3, 29, 29.3, 29.3, 29.3, 29.3, 30.6, 30.59],
        [35, 29.7, 29.4, 29.5, 29.4, 29.3, 29.3, 30.5, 30.46],
    )
)
sigma2 = sigma2

T3 = np.array([0.08, 0.17, 0.25, 0.33, 0.42, 0.50, 1.0, 2.0, T_max])
K3 = np.array([spot_price[2], 164, 172, 180])
sigma3 = np.array(
    (
        [
            volatility[2],
            volatility[2],
            volatility[2],
            volatility[2],
            volatility[2],
            volatility[2],
            volatility[2],
            volatility[2],
            volatility[2],
        ],
        [32, 29.1, 28.9, 29.3, 29.4, 29.4, 29.6, 30.6, 30.63],
        [33.6, 29.3, 29, 29.3, 29.3, 29.3, 29.3, 30.6, 30.59],
        [35, 29.7, 29.4, 29.5, 29.4, 29.3, 29.3, 30.5, 30.46],
    )
)


T4 = np.array([0.08, 0.17, 0.25, 0.33, 0.42, 0.50, 1.0, 2.0, T_max])
K4 = np.array([spot_price[3], 164, 172, 180])
sigma4 = np.array(
    (
        [
            volatility[3],
            volatility[3],
            volatility[3],
            volatility[3],
            volatility[3],
            volatility[3],
            volatility[3],
            volatility[3],
            volatility[3],
        ],
        [32, 29.1, 28.9, 29.3, 29.4, 29.4, 29.6, 30.6, 30.63],
        [33.6, 29.3, 29, 29.3, 29.3, 29.3, 29.3, 30.6, 30.59],
        [35, 29.7, 29.4, 29.5, 29.4, 29.3, 29.3, 30.5, 30.46],
    )
)

V = []  # list of variances

for i in range(N_equity):
    s_vola = globals()["sigma" + str(i + 1)]
    T_vola = globals()["T" + str(i + 1)]
    K = globals()["K" + str(i + 1)]
    V.append(
        ForwardVariance(
            reference=t,
            market_volatility_matrix=s_vola.T,
            strikes=K,
            maturity_dates=T_vola,
            strike_interp=spot_price[i],
        )
    )

plt.figure(figsize=(12, 4))
plt.title("Forward volatilities")
for i in range(N_equity):
    time = V[i].T.tolist()
    vola = V[i].forward_vol.tolist()
    time.insert(0, V[i].T[0])
    time.insert(0, 0)
    vola.insert(0, V[i].forward_vol[0])
    vola.insert(0, V[i].forward_vol[0])
    plt.step(time, vola, label="Equity " + str(i + 1), where="pre")
plt.legend()
plt.xlabel("Time t", fontsize=13)
plt.ylabel(r"$\sigma_{F}(t)$", fontsize=13)
plt.show()
# -

# ### Simulation

# %%time
maturities = np.arange(1, 31) * 0.15
model = Black(
    maturities,
    forward_curve=F,
    variance_curve=V,
    correlation_matrix=corr,
    sampling=sampling,
)
S_t = model.simulate(n_sim=N_simulation, seed=10)
G_mean = gmean(S_t, axis=2)

# +
maturities = np.arange(1, 31) * 0.15
kind = 1
f = forward_basket(F, volatility, corr, maturities)
X_t = G_mean / f
E_X = np.mean(X_t, axis=0)
K_norm = 1
pay_normalized = Vanilla_PayOff(
    X_t, K_norm, kind, sampling=sampling
)  # ATM forward pricing

"""Calculating closed form"""
call_black = GAM_Basket_option_closed_form(
    F, f, maturities, D(maturities), volatility, corr, kind
)
# -

# ### Convergence Check

n = 26
x, result, result_err = MC_Data_Blocking(pay_normalized[:, n], N_block)
mean_price = result * D(maturities[n]) * f[n]
err_price = result_err * D(maturities[n]) * f[n]

# #### In term of price

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.errorbar(x, mean_price, yerr=err_price, label="Monte Carlo")
plt.axhline(y=call_black[n], color="red", linestyle="-", label="B&S")
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
    plt.ylabel("Call option price")
elif kind == -1:
    plt.ylabel("Put option price")
plt.show()

# +
z = np.std(pay_normalized.T[n]) * f[n] * D(maturities[n])
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

font = 12
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.title(
    "log$(K/F)$ = "
    + str(log(K_norm))
    + " -- Maturity = "
    + str(round(maturities[n], 1))
    + " [yr]",
    fontsize=font,
)
plt.errorbar(x, imp_volatility_mean, yerr=[y_lower, y_upper], label="MC pricing")
plt.axhline(y=sigma_basket, lw=2, c="xkcd:red", linestyle="-.", label="BS model")
plt.xlabel("# MC throws", fontsize=font)
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
plt.grid(True)
plt.ylabel("IV", fontsize=font)
plt.legend()
plt.show()

# ### Implied volatility in function of Log-moneyness log(K/F)

# +
# %%time
logmoneyness = np.arange(-10, 11) * 0.1
imp_volatility = np.zeros((len(logmoneyness), len(maturities)))
imp_volatility_plus = np.zeros((len(logmoneyness), len(maturities)))
imp_volatility_minus = np.zeros((len(logmoneyness), len(maturities)))
for i in range(len(logmoneyness)):
    if logmoneyness[i] >= 0.0:
        kind = 1
    elif logmoneyness[i] < 0.0:
        kind = -1
    K = exp(logmoneyness[i])
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
plt.suptitle(
    "Geometric Average Basket Option: Implied volatility at fixed log-moneyness",
    fontsize=18,
)
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
plt.title("log(K/F): " + str(round(logmoneyness[num[i]], c)), fontsize=13)
plt.axhline(
    y=sigma_basket, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
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
plt.title("log(K/F): " + str(round(logmoneyness[num[i]], c)), fontsize=13)
plt.axhline(
    y=sigma_basket, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
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
plt.title("log(K/F): " + str(round(logmoneyness[num[i]], c)), fontsize=13)
plt.axhline(
    y=sigma_basket, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("Maturity", fontsize=13)
plt.ylabel("IV", fontsize=13)
# plt.ylim(0.05,0.25)
plt.legend()
plt.show()

# +
plt.figure(figsize=(22, 13))
plt.suptitle(
    "Geometric Average Basket Option: Implied volatility at fixed maturity", fontsize=18
)
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
    y=sigma_basket, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("log(K/F)", fontsize=13)
plt.ylabel("IV", fontsize=13)
plt.title("Maturity: " + str(round(maturities[num[i]], c)))
# plt.ylim(0.193,0.205)
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
    y=sigma_basket, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
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
    y=sigma_basket, color="black", alpha=0.5, linestyle="--", label="B&S volatility"
)
plt.xlabel("log(K/F)", fontsize=13)
plt.ylabel("IV", fontsize=13)
plt.title("Maturity: " + str(round(maturities[num[i]], c)))
plt.legend()
plt.show()
