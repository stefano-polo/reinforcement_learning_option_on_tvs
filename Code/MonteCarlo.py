import numpy as np

def error(AV,AV2,n,l):  # Function for statistical uncertainty estimation
    if n==0:
        return 0
    else:
        return np.sqrt((AV2[n][l] - AV[n][l]**2)/n)


def Monte_Carlo_Analisys(data,N_block):
    N = N_block
    M= int(len(data))              # Total number of throws
    L=int(M/N)           # Number of throws in each block, please use for M a multiple of N
    K = int(len(data.T))
    ave = np.zeros((N,K))
    av2 = np.zeros((N,K))
    sum_prog = np.zeros((N,K))
    su2_prog = np.zeros((N,K))
    err_prog = np.zeros((N,K))
    x = np.arange(N)
    x*=L
    for l in range(K):
        for i in range(N):
            sum = 0
            for j in range(L):
                k = j+i*L
                sum += data[k][l]
            ave[i][l] = sum/L       # r_i
            av2[i][l] = (ave[i][l])**2 # (r_i)^2
    for l in range (K):
        for i in range(N):
            for j in range(i+1):
                sum_prog[i][l] += ave[j][l] # SUM_{j=0,i} r_j
                su2_prog[i][l] += av2[j][l] # SUM_{j=0,i} (r_j)^2
            sum_prog[i][l]/=(i+1) # Cumulative average
            su2_prog[i][l]/=(i+1) # Cumulative square average
            err_prog[i][l] = error(sum_prog,su2_prog,i,l) # Statistical uncertainty


    return x,sum_prog,err_prog


def Monte_Carlo_Result(data,N_block):
    N = N_block
    M= int(len(data))              # Total number of throws
    L=int(M/N)           # Number of throws in each block, please use for M a multiple of N
    K = int(len(data.T))
    ave = np.zeros((N,K))
    av2 = np.zeros((N,K))
    sum_prog = np.zeros((N,K))
    su2_prog = np.zeros((N,K))
    err_prog = np.zeros((N,K))
    for l in range(K):
        for i in range(N):
            sum = 0
            for j in range(L):
                k = j+i*L
                sum += data[k][l]
            ave[i][l] = sum/L       # r_i
            av2[i][l] = (ave[i][l])**2 # (r_i)^2
    for l in range (K):
        for i in range(N):
            for j in range(i+1):
                sum_prog[i][l] += ave[j][l] # SUM_{j=0,i} r_j
                su2_prog[i][l] += av2[j][l] # SUM_{j=0,i} (r_j)^2
            sum_prog[i][l]/=(i+1) # Cumulative average
            su2_prog[i][l]/=(i+1) # Cumulative square average
            err_prog[i][l] = error(sum_prog,su2_prog,i,l) # Statistical uncertainty

    mean = sum_prog[len(sum_prog)-1]
    err = err_prog[len(err_prog)-1]
    return mean,err
