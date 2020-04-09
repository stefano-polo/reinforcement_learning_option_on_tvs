import numpy as np
import time, sys
import pricing as p
import MonteCarlo as MC


Nsim = int(1e6)
volatility = 0.25
spot = 10
npoints = 150
strike = np.linspace(0,200,npoints)
maturities = np.linspace(0.0001,2,npoints)


r_dates = np.linspace(0,7,10)
r = np.ones(10)*0.1
discounting_curve = p.DiscountingCurve(None,r,r_dates)
forward = p.EquityForwardCurve(spot,None,discounting_curve)

bs = p.Black(volatility,forward,maturities)
bs.simulate(Nsim)
X_T = np.mean(bs.martingale,axis=0)


implied_volatility = np.zeros((npoints,npoints))
start_time = time.time()
for i in range(npoints):
    bs.Call_PayOff(strike[i])
    print(i,"Calculating strike: ",strike[i])
    mean, err = MC.Monte_Carlo_Result(bs.pay,100)
    for j in range (npoints):
        implied_volatility[i][j] = bs.newton_vol_call_div(X_T, mean, j)


print("Tempo di esecuzione cella: ---%s minuti ---" %((time.time()-start_time)/60))



np.savetxt("implied_volatility_"+str(npoints)+".txt",implied_volatility)
K, T = np.meshgrid(strike,maturities)
np.savetxt("K_matrix_for"+str(npoints),K)
np.savetxt("T_matrix_for"+str(npoints),T)
