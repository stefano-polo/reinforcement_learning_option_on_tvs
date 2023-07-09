import sys
import warnings
from typing import List, Tuple, Union

sys.path.insert(1, "./src")

import numpy as np

from pricing.pricing import (
    DiscountingCurve,
    EquityForwardCurve,
    ForwardVariance,
    LocalVolatilityCurve,
)


def LoadFromTxt(
    asset_names: Union[List, Tuple],
    folder: str = None,
    strike_interpolation_rule: str = "ATM_SPOT",
    local_vol_model: bool = False,
) -> Tuple:
    """
    Loads discounting, equity forward, correlation matrix and local volatility curves from txt files insider a folder.
    :param asset_names (Union[List, Tuple]): list of the assets whose curves are to be loaded.
    :param folder (optional str): folder where the txt files are located.
    :param strike_interpolation_rule (str): interpolation rule for market volatilities along strike direction (available: ATM_SPOT and ATM_FWD).
    :param local_vol_model (bool): if True, local volatility model is loaded.
    :return (tuple): tuple containing the discounting curve, equity forward curve, correlation matrix and local volatility curve (if local_vol_model is True).
    """

    if folder is not None:
        if folder[-1] != "/":
            folder += "/"
    else:
        folder = "./"

    dates_discounts, discounts = np.loadtxt(folder + "discount_data.txt")
    D = DiscountingCurve(
        reference=0,
        discounts=discounts,
        discount_dates=dates_discounts,
        day_count_convention=None,
    )
    F, V = [], []
    if local_vol_model:
        LV = []

    # Load correlation matrix
    if len(asset_names) > 1:
        try:
            correlation_matrix = np.loadtxt(folder + "correlation_data.txt")
        except:
            warnings.warn("Correlation matrix not found. Setting it to identity.")
            correlation_matrix = np.eye(
                len(asset_names)
            )  # if no correlation matrix is provided, use the identity matrix

    i = 0
    for name in asset_names:
        spot = np.loadtxt(folder + "spot_data" + name + ".txt")[0]
        repo_dates, repo_rates = np.loadtxt(folder + "repo_data_" + str(name) + ".txt")
        F.append(
            EquityForwardCurve(
                reference=0,
                spot=spot,
                discounting_curve=D,
                repo_rates=repo_rates,
                repo_dates=repo_dates,
                asset_name=name,
                day_count_convention=None,
            )
        )
        spot_vola = np.loadtxt(folder + "vola_data_" + str(name) + ".txt")
        vola_strikes = np.loadtxt(folder + "strikes_vola_data_" + str(name) + ".txt")
        vola_dates = np.loadtxt(folder + "maturities_vola_data_" + name + ".txt")
        if strike_interpolation_rule == "ATM_SPOT":
            strike_interpolator = spot
        elif strike_interpolation_rule == "ATM_FWD":
            strike_interpolator = F[i]
        else:
            raise ValueError(
                "Unknown interpolation rule: available are ATM_SPOT and ATM_FWD"
            )
        V.append(
            ForwardVariance(
                reference=0,
                market_volatility_matrix=spot_vola,
                strikes=vola_strikes,
                maturity_dates=vola_dates,
                strike_interp=strike_interpolator,
                day_count_convention=None,
                asset_name=name,
            )
        )
        if local_vol_model:
            try:
                lv_pars = np.loadtxt(folder + "LV_param_data_" + str(name) + ".txt")
                lv_money = np.loadtxt(
                    folder + "LV_money_vola_data_" + str(name) + ".txt"
                )
                lv_dates = np.loadtxt(
                    folder + "LV_maturities_vola_data_" + name + ".txt"
                )
                LV.append(
                    LocalVolatilityCurve(
                        lv_pars,
                        lv_money,
                        lv_dates,
                        name,
                        log_money_interpolation_rule="kruger",
                    )
                )
            except:
                warnings.warn("Local volatility curve not found for asset: " + name)
                LV.append(None)
        i += 1
    if len(asset_names) > 1:
        if local_vol_model:
            return D, F, V, LV, correlation_matrix
        else:
            return D, F, V, correlation_matrix
    else:
        if local_vol_model:
            return D, F[0], V[0], LV[0]
        else:
            return D, F[0], V[0]
