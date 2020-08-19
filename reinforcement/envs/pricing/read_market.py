import xml.etree.ElementTree as ET
from envs.pricing.pricing import EquityForwardCurve, DiscountingCurve, ForwardVariance
from numpy import array, delete, zeros, reshape, append, max


class MarketDataReader:

    def __init__(self,filename = None):
        tree = ET.parse(filename)
        self.root = tree.getroot()

    def get_stock_number(self):
        return len(self.root[3][0][1][1][1][0])

    def get_stock_names(self):
        N_stocks = self.get_stock_number()
        names = array([])
        for i in range(N_stocks):
            names=append(names,self.root[3][0][1][1][1][0][i].text)
        delete_equity = [0,11,12,13,14,15]
        return delete(names,delete_equity)

    def get_correlation(self):
        N_stocks = self.get_stock_number()
        correlation_matrix = zeros(N_stocks**2)
        for i in range (N_stocks**2):
            correlation_matrix[i] = float(self.root[3][0][1][1][0][0][i].text)
        correlation_matrix = reshape(correlation_matrix, (N_stocks,N_stocks))
        """Delete cash bank"""
        delete_equity = [0,11,12,13,14,15]
        correlation_matrix = delete(correlation_matrix,delete_equity, axis = 0)
        correlation_matrix = delete(correlation_matrix,delete_equity, axis = 1)
        return correlation_matrix

    def get_discounts(self):
        discounts = zeros(len(self.root[3][1][0][0][0][0][0]))
        discounts_dates = zeros(len(self.root[3][1][0][0][0][0][0]))
        for i in range(len(self.root[3][1][0][0][0][0][0])):
            discounts_dates[i] = float(self.root[3][1][0][0][0][0][0][i].text)
            discounts[i] = float(self.root[3][1][0][0][0][1][0][i].text)
        return DiscountingCurve(reference=self.get_reference_date(), discounts=discounts, dates=discounts_dates, act="365")

    def get_reference_date(self):
        return float(self.root[3][1][0][0][2][0].text)

    def get_spot_prices(self):
        index_equity = [1,2,3,4,5,6,7,8,9,10]
        spot_prices = zeros(len(index_equity))
        j = 0
        for i in index_equity:
            if i>=6:
                spot_prices[j] = float(self.root[3][i+3][0][1][0][0].text)
            else:
                spot_prices[j] = float(self.root[3][3+i][0][0][1][0].text)
            j = j+1
        return spot_prices

    def get_forward_curves(self):
        discountingcurve = self.get_discounts()
        index_equity = [1,2,3,4,5,6,7,8,9,10]
        F = []
        max_dates = array([])
        index = 0
        spot_prices = self.get_spot_prices()
        reference_date = self.get_reference_date()
        for i in index_equity:
            if i>=6:
                repo_dates = array([reference_date+1, max(max_dates)])
                repo_rates = zeros(2)
                F.append(EquityForwardCurve(reference=reference_date, discounting_curve=discountingcurve, repo_dates=repo_dates,repo_rates=repo_rates, spot=spot_prices[index],act="360"))
            else:
                repo_dates = zeros(len(self.root[3][3+i][0][0][0][0][0]))
                repo_rates = zeros(len(self.root[3][3+i][0][0][0][1][0]))
                for j in range (len(self.root[3][3+i][0][0][0][0][0])):
                    repo_dates[j] = float(self.root[3][3+i][0][0][0][0][0][j].text)
                    repo_rates[j] = float(self.root[3][3+i][0][0][0][1][0][j].text)
                max_dates = append(max_dates,max(repo_dates))
                F.append(EquityForwardCurve(reference=reference_date, discounting_curve=discountingcurve, repo_dates=repo_dates,repo_rates=repo_rates, spot=spot_prices[index],act="360"))
            index = index+1
        return F

    def get_volatilities(self):
        V = []
        index = 0
        forward_curve = self.get_forward_curves()
        reference_date = self.get_reference_date()
        index_equity = [1,2,3,4,5,6,7,8,9,10]
        for i in index_equity:
            vola_dates = zeros(len(self.root[3][i+3][1][2][0][0]))
            vola_strikes = zeros(len(self.root[3][i+3][1][2][2][0]))
            for j in range(len(self.root[3][i+3][1][2][0][0])):
                vola_dates[j] = float(self.root[3][i+3][1][2][0][0][j].text)
            for j in range(len(self.root[3][i+3][1][2][2][0])):
                vola_strikes[j] = float(self.root[3][i+3][1][2][2][0][j].text)
            spot_volatilities = zeros(len(self.root[3][i+3][1][2][0][0])*len(self.root[3][i+3][1][2][2][0]))
            for k in range (len(spot_volatilities)):
                spot_volatilities[k] = float(self.root[3][i+3][1][2][1][0][k].text)
            spot_volatilities = reshape(spot_volatilities,(len(vola_dates),len(vola_strikes)))
            V.append(ForwardVariance(reference=reference_date, spot_volatility=spot_volatilities, maturities=vola_dates, strikes=vola_strikes, strike_interp=forward_curve[index],act="365"))
            index = index+1
        return V
