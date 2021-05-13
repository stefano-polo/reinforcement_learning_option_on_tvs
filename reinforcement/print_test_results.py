import matplotlib.pyplot as plt 
import numpy as np

### RL Test Result ####
MC_price = 
MC_error =  

### BS PRICE ###
BS_Price = 0.040947694174563704 
BS_Price_error = 4.325646589751639e-05

### Baseline Price ###
Baseline_price = 0.04174359990484794
Baseline_price_error = 5.798528935881833e-05 

n_sigma = 2.55
MC_error = MC_error * n_sigma
BS_Price_error = BS_Price_error * n_sigma
Baseline_price_error = Baseline_price_error * n_sigma

errors = np.array([BS_Price_error,Baseline_price_error,MC_error])
prices = err = np.array([BS_Price,Baseline_price,MC_price])
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plt.xticks([1, 2, 3], ["Black ","Baseline", "RL"],rotation=20)  # Set text labels and properties.
plt.errorbar([1, 2, 3],prices,errors,fmt='o')
plt.title("Pricing Results")
#plt.savefig("strategy_comparison.pdf")
plt.show()