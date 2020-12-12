from mpi4py import MPI
import numpy as np
from scipy.interpolate import interp1d
from time import time
#import utilities
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#import generator
import xml.etree.ElementTree as ET
from pricing import LocalVolatilityCurve
from read_market import MarketDataReader
from montecarlo import MC_Analisys, MC
import lets_be_rational.LetsBeRational as lbr
from pricing import LocalVolatilityCurve, EquityForwardCurve, ForwardVariance, DiscountingCurve,piecewise_function,Vanilla_PayOff,PricingModel,LV_model



comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
for i in range(9):
    if rank == 0:
        t_start = MPI.Wtime()
    reader = MarketDataReader("TV_example.xml")
    correlation_matrix = reader.get_correlation()
    N_equity = len(correlation_matrix)
    names = reader.get_stock_names()
    spot_prices = reader.get_spot_prices()
    D = reader.get_discounts()
    F = reader.get_forward_curves()
    V = reader.get_volatilities()


    tree = ET.parse('calibration_output.xml')
    root = tree.getroot()
    N_equity = i
    name=root[1][2][1][0][N_equity].text
    expiry_yrf = 4
    n_expiries = len(root[1][2][2][N_equity][expiry_yrf][0])#.attrib
    expiries = np.array([])
    for i in range(n_expiries):
        expiries = np.append(expiries,float(root[1][2][2][N_equity][expiry_yrf][0][i].text))
    print(name)
    moneyness = 6
    moneyness_matrix = np.array([])
    n_matrix =len(root[1][2][2][N_equity][moneyness][0])
    n_strikes = int(n_matrix/n_expiries)
    for i in range(n_matrix):
        moneyness_matrix = np.append(moneyness_matrix,float(root[1][2][2][N_equity][moneyness][0][i].text))
    moneyness_matrix = moneyness_matrix.reshape(n_strikes,n_expiries)

    vola = 7
    vola_matrix = np.array([])
    for i in range(n_matrix):
        vola_matrix = np.append(vola_matrix,float(root[1][2][2][N_equity][vola][0][i].text))
    vola_matrix = vola_matrix.reshape(n_strikes,n_expiries)
    idx = np.where(names==name)[0][0]
    V = V[idx]
    F = F[idx]
    LV_curve = LocalVolatilityCurve(vola_matrix,moneyness_matrix,expiries,name)
    N_intervals = 2000
    model = LV_model(fixings=expiries, local_vol_curve=LV_curve, forward_curve=F, N_grid = N_intervals)
    N_simulation = int(2.5e5)
    gen = np.random
    gen.seed(rank)
    logX_t = model.simulate(random_gen = gen, Nsim=N_simulation, normalization=1)
    matrix = comm.gather(logX_t,root=0)
    if rank ==0:
        matrix_results = matrix[0]
        for i in range(1,size):
            matrix_results = np.append(matrix_results,matrix[i],axis=0)
        np.save(name+'_LV_1e6'+str(N_intervals),matrix_results)
        t_finish = MPI.Wtime()
        print('Wall time ',(t_finish-t_start)/60,' min')