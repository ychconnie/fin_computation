import math as math
import scipy.stats as stats

#input set up
S0 = float(input("S0:"))
r = float(input("r:"))
q = float(input("q:"))
sigma = float(input("sigma:"))
T = float(input("T:"))
K1 = float(input("K1:"))
K2 = float(input("K2:"))
K3 = float(input("K3:"))
K4 = float(input("K4:"))

d10 = (math.log(S0/K1)+(r-q-(sigma**2)/2)*T)/(sigma*T**(1/2))
d11 = (math.log(S0/K1)+(r-q+(sigma**2)/2)*T)/(sigma*T**(1/2))
d20 = (math.log(S0/K2)+(r-q-(sigma**2)/2)*T)/(sigma*T**(1/2))
d21 = (math.log(S0/K2)+(r-q+(sigma**2)/2)*T)/(sigma*T**(1/2))
d30 = (math.log(S0/K3)+(r-q-(sigma**2)/2)*T)/(sigma*T**(1/2))
d31 = (math.log(S0/K3)+(r-q+(sigma**2)/2)*T)/(sigma*T**(1/2))
d40 = (math.log(S0/K4)+(r-q-(sigma**2)/2)*T)/(sigma*T**(1/2))
d41 = (math.log(S0/K4)+(r-q+(sigma**2)/2)*T)/(sigma*T**(1/2))

pay1 = S0 * math.e**((r-q)*T) * (stats.norm(0,1).cdf(d11) - stats.norm(0,1).cdf(d21))
pay2 = -K1 * (stats.norm(0,1).cdf(d10) - stats.norm(0,1).cdf(d20))
pay3 = (K2-K1) * (stats.norm(0,1).cdf(d20) - stats.norm(0,1).cdf(d30))
pay4 = K4 * (K2-K1)/(K4-K3) * (stats.norm(0,1).cdf(d30) - stats.norm(0,1).cdf(d40))
pay5 = -(K2-K1)/(K4-K3) * S0 * math.e**((r-q)*T) * (stats.norm(0,1).cdf(d31) - stats.norm(0,1).cdf(d41))
Option_Value = (pay1 + pay2 + pay3 + pay4 + pay5) * math.e**(-r*T)
print("Option_Value:", Option_Value)

# monte-carlo simulation: run the model 20 times
from numpy.lib.function_base import average
import numpy as np
import math as math

#compute mean and sd
mu = math.log(S0) + (r-q-(sigma**2)/2) * T
dev = (T**(1/2)) * sigma

#simulate the model 20 times
payoff20 = []
count = 0
for count in range(0,20,1):
  logsample = np.random.normal(mu, dev, 10000)
  sample = []
  payoff = []
  for i in logsample:
    sample.append(math.e**i)
  for ST in sample:
    if K1 <= ST < K2:
      payoff.append(ST-K1)
    elif K2 <= ST < K3:
      payoff.append(K2-K1)
    elif K3 <= ST < K4:
      payoff.append(((K2-K1)/(K4-K3))*(K4-ST))
    else:
      payoff.append(0)
  payoff20.append(average(payoff)*math.e**(-r*T))

#derive 95% confidence level
mean = average(payoff20)
sd = np.std(payoff20)
print("95% confidence interval:["+str(mean-2*sd)+","+str(mean+2*sd)+"]")