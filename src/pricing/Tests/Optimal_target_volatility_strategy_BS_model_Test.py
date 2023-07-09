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

# # <span style="color:brown"> Markowitz solution: Evaluation of the most constervative call option price under Black and Scholes model</span>

# +
import sys

sys.path.insert(1, "../")

import time

import matplotlib.pyplot as plt
import numpy as np
from closedforms import Price_to_BS_ImpliedVolatility
from matplotlib import ticker
from montecarlo import MC_Data_Blocking
from read_market import LoadFromTxt
from targetvol import (
    CholeskyTDependent,
    Drift,
    Strategy,
    TargetVolatilityStrategy,
    TVSForwardCurve,
    time_grid_union,
)

from pricing import (
    DiscountingCurve,
    EquityForwardCurve,
    ForwardVariance,
    Vanilla_PayOff,
)

# -

# ## Parsing market data

idx = 0
folder = "../../market_data/Basket10Assets"
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
    asset_names_list, folder, strike_interpolation_rule="ATM_FWD", local_vol_model=False
)
assert len(F) == len(V) == len(correlation_matrix) == len(correlation_matrix.T)
names = []
spot_prices = []
N_equity = len(F)
for i in range(len(V)):
    assert V[i].asset_name == F[i].asset_name
    names.append(F[i].asset_name)
    spot_prices.append(F[i].spot)

# #### Discounting curve
#

# +
plt.figure(figsize=(15, 4))

plt.subplot(1, 2, 1)
x = np.linspace(0, 60, 1000)
plt.step(x, D.R(x), color="red")
plt.xlabel("Time [yr]")
plt.ylabel("zero interest rate")

plt.subplot(1, 2, 2)
plt.plot(x, D(x))
plt.xlabel("Time [yr]")
plt.ylabel("Discounting curve")
plt.show()
# -

# #### Equity spot prices

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plt.xticks(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], names, rotation=20
)  # Set text labels and properties.
plt.scatter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], spot_prices)
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.yaxis.set_major_formatter(formatter)
plt.title("Spot prices", fontsize=13)
plt.ylabel("$S_0$ [EUR]", fontsize=13)
plt.show()
print("Spot prices: ", spot_prices)

# ## Preparing the temporal structure for volatilities and EA

mu = Drift(F)
nu = CholeskyTDependent(V, np.linalg.cholesky(correlation_matrix))

# +
dimension_points = 12
plt.figure(figsize=(10, 6))
for i in range(N_equity):
    time = np.array([])
    repo = np.array([])
    time = np.append(time, F[i].T)
    repo = np.append(repo, F[i].q_values)
    time = np.append(time, nu.T[len(nu.T) - 1])
    repo = np.append(repo, F[i].q_values[len(repo) - 1])
    plt.step(time, repo, label=names[i])

plt.legend()
plt.xlim(0, nu.T[len(nu.T) - 1])
plt.title("Instant repo rates", fontsize=13)
plt.xlabel("Time [yr]", fontsize=13)
plt.ylabel(r"$\mu$(t)", fontsize=13)
# plt.savefig("Istant_repo_rates.pdf")
plt.show()
# -

plt.figure(figsize=(12, 8))
plt.title("Forward volatilities", fontsize=13)
for i in range(N_equity):
    time = V[i].T.tolist()
    vola = V[i].forward_vol.tolist()
    time.insert(0, V[i].T[0])
    time.insert(0, 0)
    time.insert(len(time) - 1, nu.T[len(nu.T) - 1])
    vola.insert(0, V[i].forward_vol[0])
    vola.insert(0, V[i].forward_vol[0])
    vola.insert(len(vola) - 1, V[i].forward_vol[len(V[i].forward_vol) - 1])
    plt.step(time, vola, label=names[i], where="pre")
plt.legend()
plt.xlabel("Time [yr]", fontsize=13)
plt.ylabel(r"$\sigma_{F}(t)$", fontsize=13)
plt.xlim(0, nu.T[len(nu.T) - 1])
# plt.savefig("forward_volatilities.pdf")
plt.show()

# ## Monte Carlo Simulation

"""Monte carlo simulation parameters"""
I_0 = 1.0
vol_target = 5.0 / 100.0
print("Target volatility: ", vol_target)
K = I_0  # ATM pricing
simulations = 1e6
N_block = 100
kind = 1  # call option
maturity_idx_valuation = -7
sampling = "antithetic"

# ### Unconstrained optimization for the allocation strategy: markowitz solution

alpha = Strategy()
alpha.Mark_strategy(mu=mu, nu=nu)
dates = alpha.T
plt.figure(figsize=(10, 6))
plt.title("Optimal strategy", fontsize=13)
for i in range(N_equity):
    plt.step(alpha.T, alpha.alpha_t.T[i], label=names[i])
plt.legend()
plt.xlabel("Time [yr]", fontsize=13)
plt.ylabel(r"$\alpha(t)$", fontsize=13)
plt.xlim(0, alpha.T[len(alpha.T) - 1])
plt.show()

# %%time
n = maturity_idx_valuation
TVSF = TVSForwardCurve(
    reference=0.0,
    vola_target=vol_target,
    spot_price=I_0,
    strategy=alpha,
    mu=mu,
    nu=nu,
    discounting_curve=D,
)
TVS = TargetVolatilityStrategy(dates, forward_curve=TVSF, sampling=sampling)
I_t = TVS.simulate(n_sim=simulations, seed=7)
pay = Vanilla_PayOff(I_t, K, kind, sampling=sampling) * D(dates)
x, result_mark, result_err_mark = MC_Data_Blocking(pay[:, n], N_block)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.title("Maturity = " + str(round(dates[n], 2)))
plt.errorbar(x, result_mark, yerr=result_err_mark)
plt.xlabel("#throws")
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
if kind == 1:
    plt.ylabel("Call option price [EUR]")
if kind == -1:
    plt.ylabel("Put option price [EUR]")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.title("Maturity = " + str(round(dates[n], 2)))
plt.errorbar(x, result_mark, yerr=result_err_mark)
plt.xlabel("#throws")
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
if kind == 1:
    plt.ylabel("Call option price [EUR]")
if kind == -1:
    plt.ylabel("Put option price [EUR]")
plt.show()

# +
forward = TVS.forward_values
X_t = I_t / forward
N_block = 100
pay_normalized = Vanilla_PayOff(X_t, K / forward, kind, sampling)
x, result, result_err = MC_Data_Blocking(pay_normalized[:, n], N_block)
imp_volatility_mean = np.zeros(N_block)
imp_volatility_plus = np.zeros(N_block)
imp_volatility_minus = np.zeros(N_block)
for i in range(N_block):
    imp_volatility_mean[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i], kind, 1.0
    )
    imp_volatility_plus[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i] + result_err[i], kind, 1.0
    )
    imp_volatility_minus[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i] - result_err[i], kind, 1.0
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
plt.title("Maturity = " + str(round(dates[n], 2)))
plt.errorbar(x, imp_volatility_mean, yerr=[y_lower, y_upper], label="Monte Carlo")
plt.axhline(y=vol_target, color="red", label="Target volatility")
plt.xlabel("#throws")
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.ylabel("IV")
plt.legend()
plt.show()

# ### Constrained optimization for the allocation strategy:
# ### $$\sum_i \alpha_i = 1\quad \text{with}\quad \alpha_i>0$$

# %%time
alpha.optimization_constrained(mu=mu, nu=nu, n_trial=100)
plt.figure(figsize=(10, 6))
plt.title("Optimal strategy", fontsize=13)
for i in range(N_equity):
    plt.step(alpha.T, alpha.alpha_t.T[i], label=names[i])
plt.legend()
plt.xlabel("Time [yr]", fontsize=13)
plt.ylabel(r"$\alpha(t)$", fontsize=13)
plt.xlim(0, alpha.T[len(alpha.T) - 1])
plt.show()

# %%time
TVSF = TVSForwardCurve(
    reference=0.0,
    vola_target=vol_target,
    spot_price=I_0,
    strategy=alpha,
    mu=mu,
    nu=nu,
    discounting_curve=D,
)
TVS = TargetVolatilityStrategy(dates, forward_curve=TVSF, sampling=sampling)
I_t = TVS.simulate(n_sim=simulations, seed=11)
pay = Vanilla_PayOff(I_t, K, kind, sampling=sampling) * D(dates)
x, result_opt1, result_err_opt1 = MC_Data_Blocking(pay[:, n], N_block)

# +
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.title("Maturity = " + str(round(dates[n], 2)))
plt.errorbar(x, result_opt1, yerr=result_err_opt1)
plt.xlabel("#throws")
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
if kind == 1:
    plt.ylabel("Call option price [EUR]")
if kind == -1:
    plt.ylabel("Put option price [EUR]")

plt.show()

# +
forward = TVS.forward_values
X_t = I_t / forward
pay_normalized = Vanilla_PayOff(X_t, K / forward, kind, sampling)
x, result, result_err = MC_Data_Blocking(pay_normalized[:, n], N_block)
imp_volatility_mean = np.zeros(N_block)
imp_volatility_plus = np.zeros(N_block)
imp_volatility_minus = np.zeros(N_block)
for i in range(N_block):
    imp_volatility_mean[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i], kind, 1.0
    )
    imp_volatility_plus[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i] + result_err[i], kind, 1.0
    )
    imp_volatility_minus[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i] - result_err[i], kind, 1.0
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
plt.title("Maturity = " + str(round(dates[n], 2)))
plt.errorbar(x, imp_volatility_mean, yerr=[y_lower, y_upper], label="Monte Carlo")
plt.axhline(y=vol_target, color="red", label="Target volatility")
plt.xlabel("#throws")
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.ylabel("IV")
plt.legend()
plt.show()

# #### This constrained strategy wins over intuitive strategies?

# %%time
alpha.Intuitive_strategy1(F, np.max(dates))
TVSF = TVSForwardCurve(
    reference=0.0,
    vola_target=vol_target,
    spot_price=I_0,
    strategy=alpha,
    mu=mu,
    nu=nu,
    discounting_curve=D,
)
TVS = TargetVolatilityStrategy(dates, forward_curve=TVSF, sampling=sampling)
I_t = TVS.simulate(n_sim=simulations, seed=7)
pay = Vanilla_PayOff(I_t, K, kind, sampling) * D(dates)
x, result_intuitive1, result_err_intuitive1 = MC_Data_Blocking(pay[:, n], N_block)

# %%time
alpha.Intuitive_strategy2(mu)
TVSF = TVSForwardCurve(
    reference=0.0,
    vola_target=vol_target,
    spot_price=I_0,
    strategy=alpha,
    mu=mu,
    nu=nu,
    discounting_curve=D,
)
TVS = TargetVolatilityStrategy(dates, forward_curve=TVSF, sampling=sampling)
I_t = TVS.simulate(n_sim=simulations, seed=10)
pay = Vanilla_PayOff(I_t, K, kind, sampling) * D(dates)
x, result_intuitive2, result_err_intuitive2 = MC_Data_Blocking(pay[:, n], N_block)

# %%time
alpha.Intuitive_strategy3(mu, nu)
TVSF = TVSForwardCurve(
    reference=0.0,
    vola_target=vol_target,
    spot_price=I_0,
    strategy=alpha,
    mu=mu,
    nu=nu,
    discounting_curve=D,
)
TVS = TargetVolatilityStrategy(dates, forward_curve=TVSF, sampling=sampling)
I_t = TVS.simulate(n_sim=simulations, seed=10)
pay = Vanilla_PayOff(I_t, K, kind, sampling=sampling) * D(dates)
x, result_intuitive3, result_err_intuitive3 = MC_Data_Blocking(pay[:, n], N_block)

# +
import seaborn as sns

sns.set_theme(style="white", context="talk")

f, ax1 = plt.subplots(1, 1, figsize=(8.5, 4), sharex=True)
x = np.array(["S$_A$", "S$_B$", "S$_C$", "BS$^*$"])
y1 = np.array(
    [
        result_intuitive1[N_block - 1],
        result_intuitive2[N_block - 1],
        result_intuitive3[N_block - 1],
        result_opt1[N_block - 1],
    ]
)
sns.barplot(x=x, y=y1, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("Option Price [EUR]")
plt.title("Maturity = " + str(round(dates[n], 2)))
plt.ylim(0.038, 0.04145)
# -

# ### Constrained optimization for the allocation strategy:
# ### $$\left|\alpha_i\right| \leq 10\% \quad \forall i$$

# +
# %%time
alpha.optimization_constrained(
    mu=mu, nu=nu, n_trial=10, long_limit=10 / 100, constraint_strategy=2
)

plt.figure(figsize=(10, 6))
plt.title("Optimal strategy", fontsize=13)
for i in range(N_equity):
    plt.step(alpha.T, alpha.alpha_t.T[i], label=names[i])
plt.legend()
plt.xlabel("Time [yr]", fontsize=13)
plt.ylabel(r"$\alpha(t)$", fontsize=13)
plt.xlim(0, alpha.T[len(alpha.T) - 1])
plt.show()
# -

# %%time
TVSF = TVSForwardCurve(
    reference=0.0,
    vola_target=vol_target,
    spot_price=I_0,
    strategy=alpha,
    mu=mu,
    nu=nu,
    discounting_curve=D,
)
TVS = TargetVolatilityStrategy(dates, forward_curve=TVSF, sampling=sampling)
I_t = TVS.simulate(n_sim=simulations, seed=7)
pay = Vanilla_PayOff(I_t, K, kind, sampling=sampling) * D(dates)
x, result_opt2, result_err_opt2 = MC_Data_Blocking(pay[:, n], N_block)

# +
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.title("Maturity = " + str(round(dates[n], 2)))
plt.errorbar(x, result_opt2, yerr=result_err_opt2)
plt.xlabel("#throws")
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
if kind == 1:
    plt.ylabel("Call option price [EUR]")
if kind == -1:
    plt.ylabel("Put option price [EUR]")

plt.show()

# +
forward = TVS.forward_values
X_t = I_t / forward
pay_normalized = Vanilla_PayOff(X_t, K / forward, kind, sampling)
x, result, result_err = MC_Data_Blocking(pay_normalized[:, n], N_block)
imp_volatility_mean = np.zeros(N_block)
imp_volatility_plus = np.zeros(N_block)
imp_volatility_minus = np.zeros(N_block)
for i in range(N_block):
    imp_volatility_mean[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i], kind, 1.0
    )
    imp_volatility_plus[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i] + result_err[i], kind, 1.0
    )
    imp_volatility_minus[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i] - result_err[i], kind, 1.0
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
plt.title("Maturity = " + str(round(dates[n], 2)))
plt.errorbar(x, imp_volatility_mean, yerr=[y_lower, y_upper], label="Monte Carlo")
plt.axhline(y=vol_target, color="red", label="Target volatility")
plt.xlabel("#throws")
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.ylabel("IV")
plt.legend()
plt.show()

# ### Constrained optimization for the allocation strategy:
# ### $$\sum_i\alpha_i \leq 90\%  \quad \forall \alpha_i>0$$
# ### $$\left|\sum_i\alpha_i \right| \leq 10\%  \quad \forall\alpha_i<0$$

# +
# %%time
alpha.optimization_constrained(
    mu=mu,
    nu=nu,
    n_trial=10,
    long_limit=90 / 100,
    short_limit=10 / 100,
    constraint_strategy=3,
)

plt.figure(figsize=(10, 6))
plt.title("Optimal strategy", fontsize=13)
for i in range(N_equity):
    plt.step(alpha.T, alpha.alpha_t.T[i], label=names[i])
plt.legend()
plt.xlabel("Time [yr]", fontsize=13)
plt.ylabel(r"$\alpha(t)$", fontsize=13)
plt.xlim(0, alpha.T[len(alpha.T) - 1])
plt.show()
# -

# %%time
TVSF = TVSForwardCurve(
    reference=0.0,
    vola_target=vol_target,
    spot_price=I_0,
    strategy=alpha,
    mu=mu,
    nu=nu,
    discounting_curve=D,
)
TVS = TargetVolatilityStrategy(dates, forward_curve=TVSF, sampling=sampling)
I_t = TVS.simulate(n_sim=simulations, seed=7)
pay = Vanilla_PayOff(I_t, K, kind, sampling=sampling) * D(dates)
x, result_opt3, result_err_opt3 = MC_Data_Blocking(pay[:, len(dates) - 1], N_block)

# +
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.title("Maturity = " + str(round(dates[n], 2)))
plt.errorbar(x, result_opt3, yerr=result_err_opt3)
plt.xlabel("#throws")
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
if kind == 1:
    plt.ylabel("Call option price [EUR]")
if kind == -1:
    plt.ylabel("Put option price [EUR]")

plt.show()

# +
forward = TVS.forward_values
X_t = I_t / forward
pay_normalized = Vanilla_PayOff(X_t, K / forward, kind, sampling)
x, result, result_err = MC_Data_Blocking(pay_normalized[:, n], N_block)
imp_volatility_mean = np.zeros(N_block)
imp_volatility_plus = np.zeros(N_block)
imp_volatility_minus = np.zeros(N_block)
for i in range(N_block):
    imp_volatility_mean[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i], kind, 1.0
    )
    imp_volatility_plus[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i] + result_err[i], kind, 1.0
    )
    imp_volatility_minus[i] = Price_to_BS_ImpliedVolatility(
        dates[n], 1.0, K / forward[n], result[i] - result_err[i], kind, 1.0
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
plt.title("Maturity = " + str(round(dates[n], 2)))
plt.errorbar(x, imp_volatility_mean, yerr=[y_lower, y_upper], label="Monte Carlo")
plt.axhline(y=vol_target, color="red", label="Target volatility")
plt.xlabel("#throws")
formatter = ticker.ScalarFormatter(useMathText=True)  # scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.ylabel("IV")
plt.legend()
plt.show()
