from pyGem import GemItem
import numpy as np
import pandas as pd
import warnings
import os

"""
This script is used to convert xml files with FE market data conventions to txt files used by this library.
The user must provide the xml file (FileName), the folder of the xml file, a list of the assets names that 
the user wants to export (asset_names), the discounting curve nave (discount_curve) and the output folder 
where the txt files will be saved (output_folder).
"""

############################################################ SCRIPT INPUTS #########################################################################################

FileName = 'calibration_output'
input_folder = "../../market_data"
asset_names = "DJ 50 TR", "S&P 500 NET EUR", "MSCI EM MKT EUR", "I NKY NTR EUR", "FTSE100 NTR E", "SMI TR EUR", "DAX 30 E", "FTSEMIBN", "CAC 40 NTR", "HSI NTR EUR"
discount_curve = "EUR"
saving_folder = "../../market_data/Basket10Assets"

######################################################################################################################################################################


if input_folder[-1] != '/':
    input_folder += '/'
if saving_folder[-1] != '/':
    saving_folder += '/'
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)

# Read GemItem
market = GemItem()
market.load("", "file:///" + str(input_folder) + FileName + ".xml")

# Read discounting curve
reference = market.get(f"market.{discount_curve}.zero_curve.{discount_curve} :STD.reference_date")
dates = np.array(market.get(f"market.{discount_curve}.zero_curve.{discount_curve} :STD.data.dates"))
discounts = np.array(market.get(f"market.{discount_curve}.zero_curve.{discount_curve} :STD.data.discounts"))
day_count_convention = market.get(f"market.{discount_curve}.zero_curve.{discount_curve} :STD.convention.day_count")

if day_count_convention == "ACT365":
    discount_dates = (dates - reference) / 365.
elif day_count_convention == "ACT360":
    discount_dates = (dates - reference) / 360.
else:
    raise ValueError(f" {day_count_convention} day count convention not supported")

np.savetxt(saving_folder + "discount_data.txt", (discount_dates, discounts))

# Read Correlation Matrix
correlation_matrix = np.array(market.get("market.data.correlation_matrix.data.correlations"))
asset_names_in_correlation = list(market.get("market.data.correlation_matrix.data.names"))
correlation_dataframe = pd.DataFrame(correlation_matrix, columns=asset_names_in_correlation,
                                     index=asset_names_in_correlation)
# delete correlation not required
for asset in asset_names_in_correlation:
    if asset not in asset_names:
        correlation_dataframe = correlation_dataframe.drop(asset, axis=0)
        correlation_dataframe = correlation_dataframe.drop(asset, axis=1)
np.savetxt(saving_folder + "correlation_data.txt", correlation_dataframe.values)

# Read Market Data and Local Volatility
for name in asset_names:
    market_volatilities = np.array(market.get("market." + name + ".volatility.data.volatilities"))
    maturities = np.array(market.get("market." + name + ".volatility.data.expiry_dates"))
    vola_reference = market.get("market." + name + ".volatility.reference_date")
    maturities = (maturities - vola_reference) / 365.
    strikes = np.array(market.get("market." + name + ".volatility.data.strikes"))
    spot = market.get("market." + name + ".forwards.data.spot")
    try:
        repo_dates = np.array(market.get("market." + name + ".forwards.data.repo.dates"))
        repo_day_count_convention = market.get("market." + name + ".forwards.convention.repo.day_count")
        repo_reference = market.get("market." + name + ".forwards.reference_date")
        if repo_day_count_convention == "ACT365":
            repo_dates = (repo_dates - repo_reference) / 365.
        elif repo_day_count_convention == "ACT360":
            repo_dates = (repo_dates - repo_reference) / 360.
        else:
            raise ValueError(f" {day_count_convention} day count convention not supported")
        repo_rates = np.array(market.get("market." + name + ".forwards.data.repo.rates"))
    except:
        warning_string = "No repo data for " + name + ". Setting repo_rates to zero"
        warnings.warn(warning_string)
        repo_dates = np.array([0., 10., 20.])
        repo_rates = np.array([0., 0., 0.])
    np.savetxt(saving_folder + "spot_data" + name + ".txt", np.append(spot, spot))
    np.savetxt(saving_folder + "repo_data_" + str(name) + ".txt", (repo_dates, repo_rates))
    np.savetxt(saving_folder + "vola_data_" + str(name) + ".txt", market_volatilities)
    np.savetxt(saving_folder + "strikes_vola_data_" + str(name) + ".txt", strikes)
    np.savetxt(saving_folder + "maturities_vola_data_" + name + ".txt", maturities)
    try:
        lv_dates = np.array(market.get("model.local_volatility.parameters." + name + ".expiry_yrf"))
        lv_money = np.array(market.get("model.local_volatility.parameters." + name + ".local_volatility_money"))
        lv_pars = np.array(market.get("model.local_volatility.parameters." + name + ".local_volatility_pars"))
        np.savetxt(saving_folder + "LV_param_data_" + str(name) + ".txt", lv_pars)
        np.savetxt(saving_folder + "LV_money_vola_data_" + str(name) + ".txt", lv_money)
        np.savetxt(saving_folder + "LV_maturities_vola_data_" + name + ".txt", lv_dates)
    except:
        warning_string = "No local volatility data for " + name
        warnings.warn(warning_string)
        pass

