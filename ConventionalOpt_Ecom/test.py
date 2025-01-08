import numpy as np

consts = np.ones(100)
P=363600
N = len(consts)

x = np.arange(3600,P,(P)/(N))

Power = []
Power.insert(0,48.6e6)
Power.insert(1,48.6e6)

for i in range(0,len(x),1):
    FFp = consts[i] + 48.6e6
    Power.append(FFp)

print(Power)
print(x)
k = len(x)

for i in range(len(x)):
    x = np.insert(x,2*i+1 , x[2*i] + 2000)
    Power.insert(2*i+1,Power[2*i])

print(Power)
print(x)

    
x = np.append(x,2e6)
Power.append(48.6e6)


x = x + 1e6
x = np.insert(x,0,0)
x = np.insert(x,1,1000000)
xt = np.atleast_2d(x)
xt = np.transpose(xt)


Power = np.array(Power)
Powert = np.atleast_2d(Power)
Powert = np.transpose(Powert)

Tup = np.append(xt,Powert, axis =1)

    
FeedForward = Tup.tolist()
print(FeedForward)