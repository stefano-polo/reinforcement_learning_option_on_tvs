import numpy as np
import lets_be_rational.LetsBeRational as lbr
from numpy import log, exp,sqrt
from scipy.interpolate import interp1d,RegularGridInterpolator
from pricing import ACT_365, DiscountingCurve, EquityForwardCurve, Curve

class LocalVolatilityCurve(Curve):
    
    def __init__(self, market_volatility=None, strikes=None, maturities=None):
        self.volatilities = market_volatility
        self.K = strikes
        self.T = np.append(0.,maturities[:-1])     #it is fundamental this transformation for the piecewise interpolation
        self.vola_interpolated = interp1d(self.K,self.volatilities,axis=0,fill_value="extrapolate")   #linear interpolation along strike
    
    def curve(self,price):
        return self.vola_interpolated(np.array(price))
    
    def value_at_time(self,time,price):
        if time in self.T:
            return self(price).T[np.searchsorted(self.T, time, side='left')]
        else:
            return self(price).T[np.searchsorted(self.T, time, side='left')-1]
    
def A_i_matrix(alpha_i, beta_i, gamma_i):
    b = np.diag(beta_i)
    a = np.diag(alpha_i[1:],k=-1)
    c = np.diag(gamma_i[:-1],k=1)
    A = a +b +c
    A[0] = 0
    A[-1] = 0
    return A


def alpha_i_vector(eta_i_plus_square, dh):
    """Calculates the alpha_i coefficients of the tri-diagonal matrix"""
    return (eta_i_plus_square) * 0.5 * (-(0.5/dh)-1/(dh**2))

def beta_i_vector(eta_i_plus_square, dh):
    """Calculates the beta_i coefficients of the tri-diagonal matrix"""
    return (eta_i_plus_square) * (1/(dh**2))


def gamma_i_vector(eta_i_plus_square, dh):
    """Calculates the gamma_i coefficients of the tri-diagonal matrix"""
    return (eta_i_plus_square) * 0.5 * ((0.5/dh)-1/(dh**2))

def backward_euler_method(c_in,t_in,t_fin,L_t,L_h,d_h,h_grid,forward, eta_curve):
    Delta_t = abs(t_in-t_fin)/L_t
    Delta_h = d_h
    for i in range(L_t):
        t2 = (i+1)*Delta_t + t_in
        curve = eta_curve.value_at_time(t2,exp(h_grid)*forward(t2))**2
        alpha_i = alpha_i_vector(curve, Delta_h)
        beta_i = beta_i_vector(curve, Delta_h)
        gamma_i = gamma_i_vector(curve, Delta_h)
        A_i = A_i_matrix(alpha_i,beta_i,gamma_i)
        c_in = np.linalg.inv((np.identity(L_h+1) + Delta_t*A_i))@c_in
    return c_in

def forward_euler_method(c_in,t_in,t_fin,L_t,L_h,d_h,h_grid,forward,eta_curve):
    Delta_t = abs(t_in-t_fin)/L_t
    Delta_h = d_h
    for i in range(L_t):
        t2 = (i)*Delta_t+t_in
        curve = eta_curve.value_at_time(t2,exp(h_grid)*forward(t2))**2
        alpha_i = alpha_i_vector(curve, Delta_h)
        beta_i = beta_i_vector(curve, Delta_h)
        gamma_i = gamma_i_vector(curve, Delta_h)
        A_i = A_i_matrix(alpha_i,beta_i,gamma_i)
        c_in = c_in - Delta_t* A_i@c_in
    return c_in


def crank_nicolson_method(c_in,t_in,t_fin,L_t,L_h,d_h,h_grid,forward,eta_curve):
    Delta_t = abs(t_in-t_fin)/L_t
    Delta_h = d_h
    for i in range(L_t):
        t2 = (i+1)*Delta_t+t_in
        t1 = i*Delta_t+t_in
        curve1 = eta_curve.value_at_time(t1,exp(h_grid)*forward(t1))**2
        curve2 = eta_curve.value_at_time(t2,exp(h_grid)*forward(t2))**2
        alpha_i = alpha_i_vector(curve1, Delta_h)
        beta_i = beta_i_vector(curve1, Delta_h)
        gamma_i = gamma_i_vector(curve1, Delta_h)
        A_i_1 = A_i_matrix(alpha_i,beta_i,gamma_i)
        alpha_i = alpha_i_vector(curve2, Delta_h)
        beta_i = beta_i_vector(curve2, Delta_h)
        gamma_i = gamma_i_vector(curve2, Delta_h)
        A_i_2 = A_i_matrix(alpha_i,beta_i,gamma_i)
        c_in = np.linalg.inv((np.identity(L_h+1) + 0.5*Delta_t*A_i_2))@(np.identity(L_h+1) - 0.5*Delta_t*A_i_1)@c_in
    return c_in

def call_options_pricer(maturities, L_t, L_h,forward, eta_curve, h_min=-4., h_max=4.):
    algorithm = crank_nicolson_method
    d_h = (h_max-h_min)/L_h
    h_grid = np.linspace(h_min+d_h,h_max,L_h,endpoint=True)
    h_grid = np.insert(h_grid,0,h_min)
    c_i = np.maximum(1-exp(h_grid),np.zeros(L_h+1))   #call option at time 0
    c_i[0] = 1-exp(h_min)
    c_i[-1] = 0
    matrix_call = np.array([])
    for i in range(len(maturities)):
        if i == 0:
            matrix_call = algorithm(c_i,0,maturities[0],L_t,L_h,d_h,h_grid, forward,eta_curve)
        if i>0:
            Delta_t = (maturities[i]-maturities[i-1])/L_t
        if i ==1:
            matrix_call = np.stack((matrix_call,algorithm(matrix_call,maturities[0],maturities[i],L_t,L_h,d_h,h_grid,forward, eta_curve)),axis=1)
        if i>1:
            matrix_call = np.insert(matrix_call,len(matrix_call.T),algorithm(matrix_call.T[i-1],maturities[i-1],maturities[i],L_t,L_h,d_h,h_grid,forward, eta_curve),axis=1)

    return matrix_call, h_grid

def from_price_to_vola(matrix_call, maturities, h):
    vola_matrix = np.zeros(matrix_call.shape)
    k = exp(h)
    for i in range(len(h)):
        for j in range(len(maturities)):
            vola_matrix[i,j] = lbr.implied_volatility_from_a_transformed_rational_guess(matrix_call[i,j],1.,k[i],maturities[j],1)
    return vola_matrix

def loss_function(iv_model, market_vola):
    return np.sum((iv_model-market_vola)**2)

def back_coordinates(IV,maturities,F,market_strikes,h_grid):
    interpo_strike = interp1d(h_grid,IV,axis=0,fill_value="extrapolate")
    IV_new_coord = np.zeros((len(market_strikes),len(maturities)))
    forward = F(maturities)
    for i in range(len(market_strikes)):
        h = np.log(market_strikes[i]/forward)
        IV_new_coord[i,:] = np.diagonal(interpo_strike(h))
    return IV_new_coord

def new_LV_points(LV_in,vola_f_LV,vola_f_market):
    new_points = LV_in * (vola_f_market/vola_f_LV)
    return new_points

   
def forward_volatility(spot_vola_matrix, maturities):
    f_vol = spot_vola_matrix.T[0]
    for i in range(1,len(maturities)):
        f_v = np.sqrt(((spot_vola_matrix.T[i]**2)*maturities[i]-(spot_vola_matrix.T[i-1]**2)*maturities[i-1])/(maturities[i]-maturities[i-1]))
        if i==1:
            f_vol = np.stack((f_vol,f_v),axis=1)
        else:
            f_vol = np.insert(f_vol,len(f_vol.T),f_v,axis=1)
    return f_vol