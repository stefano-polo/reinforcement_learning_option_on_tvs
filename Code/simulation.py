import numpy as np
from pricing import DiscountingCurve, EquityForwardCurve, Black
from montecarlo import MC


Nsim = int(1e6)
seed =10
volatility = 20/100
reference = 0
spot = 150
npoints = 100
discount = 1/100
strike = np.linspace(0,300,npoints)
maturities = np.linspace(0.0001,3,npoints)

"""Writing information file about simulation"""

file1 = open("README.txt","w")
file1.write("Black & Scholes model \n\n")
file1.write("N Monte Carlo simulation: "+str(Nsim)+"\n")
file1.write("Seed of the simulation: "+str(seed))
file1.write("Volatility: "+str(volatility)+" \n")
file1.write("Costant discount factor: "+str(discount)+"\n")
file1.write("Reference: "+str(reference)+"\n")
file1.write("Spot price: "+str(spot)+"\n")
file1.write("Number of maturities and strikes: "+str(npoints)+"\n")
file1.write("Range of maturities ["+str(maturities[0])+", "+str(maturities[npoints-1])+"] \n")
file1.write("Range of strikes ["+str(strike[0])+", "+str(strike[npoints-1])+"] \n")
file1.close()

r_dates = np.linspace(0,7,10)
r = np.ones(10)*discount
discounting_curve = DiscountingCurve(reference,r,r_dates)
forward = EquityForwardCurve(spot,reference,discounting_curve)

print("Martingale: Start \n")
bs = Black(volatility,forward,maturities)
bs.setSeed(seed)
bs.simulate(Nsim)
X_T = np.mean(bs.martingale,axis=0)
print("Martingale: Done \n")
print("Saving martingale")
#np.savetxt("Martingale_file_matrix.txt",bs.martingale)
print("Martingale saved")

implied_volatility_mean = np.zeros((npoints,npoints))
implied_volatility_plus = np.zeros((npoints,npoints))
implied_volatility_minus = np.zeros((npoints,npoints))

for i in range(npoints):
    bs.Call_PayOff(strike[i])
    mean, err = MC(bs.pay)
    plus = mean+err
    minus = mean-err
    print(i,"Calculating strike: ",strike[i])
    for j in range (npoints):
        implied_volatility[i][j] = bs.newton_implied_volatility(X_T, mean, j)
        implied_volatility_plus[i][j] = bs.newton_implied_volatility(X_T, plus, j)
        implied_volatility_minus[i][j] = bs.newton_implied_volatility(X_T, minus, j)


print("Saving_Files...")
K, T = np.meshgrid(strike,maturities)
np.savetxt("mean_implied_volatility_for_nKT"+str(npoints)+"_nMC"+str(Nsim)+".txt",implied_volatility)    #saving results
np.savetxt("plus_implied_volatility_for_nKT"+str(npoints)+"_nMC"+str(Nsim)+".txt",implied_volatility_plus)    #saving results
np.savetxt("minus_implied_volatility_for_nKT"+str(npoints)+"_nMC"+str(Nsim)+".txt",implied_volatility_minus)    #saving results
np.savetxt("K_matrix_for_nKT"+str(npoints)+"_nMC"+str(Nsim)+".txt",K)
np.savetxt("T_matrix_for_nKT"+str(npoints)+"_nMC"+str(Nsim)+".txt",T)
print("COMPLETED")
