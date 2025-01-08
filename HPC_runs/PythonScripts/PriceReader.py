import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def importdict(filename, zone, offset):#creates a function to read the csv
    times_seconds = []
    prices = []
    #create data frame from csv with pandas module
    df=pd.read_csv(filename+'.csv', names=['Date', 'price', 'zone'],sep=',',parse_dates=[0],skiprows=1) #or:, infer_datetime_format=True)
    df['diff'] = df['Date'] - df['Date'].iloc[0]
    df['Times'] = df['diff'].dt.total_seconds()
    fileDATES=df.T.to_dict().values()#export the data frame to a python dictionary
    fileDATES = list(fileDATES)
    for line in fileDATES:
        if line['zone'] == zone:
            times_seconds.append(int(line['Times'])+offset)
            prices.append(line['price'])

    return times_seconds, prices #return list of times in seconds and prices

def importdict2(filename, offset, fileyear):#creates a function to read the csv
    times_seconds = []
    prices = []
    #create data frame from csv with pandas module
    df=pd.read_csv(filename+str(fileyear)+'.csv', names=['hour', 'price'],sep=',',skiprows=1) #or:, infer_datetime_format=True)
    fileDATES=df.T.to_dict().values()#export the data frame to a python dictionary
    fileDATES = list(fileDATES)
    shortdict = fileDATES[offset:offset+200]
    for line in shortdict:
        times_seconds.append((int(line['hour']) - int(shortdict[0]['hour']))*3600)
        prices.append(line['price'])

    return times_seconds, prices #return list of times in seconds and prices

def importdict3(filename, offset, fileyear):#creates a function to read the csv
    times_seconds = []
    prices = []
    DNIs = []
    IAMs = []
    #create data frame from csv with pandas module
    df=pd.read_csv(filename+str(fileyear)+'.csv', names=['Year', 'Month', 'Day', 'Hour', 'Minute', 'DNI', 'Price', 'IAM'],sep=',',skiprows=1) #or:, infer_datetime_format=True)
    fileDATES=df.T.to_dict().values()#export the data frame to a python dictionary
    fileDATES = list(fileDATES)
    shortdict = fileDATES[offset:offset+150]
    j=0
    for line in shortdict:
        times_seconds.append(j*3600)
        prices.append(line['Price'])
        DNIs.append(line['DNI'])
        IAMs.append(line['IAM'])
        j = j + 1

    return times_seconds, prices, DNIs, IAMs #return list of times in seconds and prices

def importdict4(filename, offset, fileyear):#creates a function to read the csv
    times_seconds = []
    prices = []
    DNIs = []
    IAMs = []
    #create data frame from csv with pandas module
    df=pd.read_csv(filename+str(fileyear)+'.csv', names=['Year', 'Month', 'Day', 'Hour', 'Minute', 'DNI', 'Price', 'IAM'],sep=',',skiprows=1) #or:, infer_datetime_format=True)
    fileDATES=df.T.to_dict().values()#export the data frame to a python dictionary
    fileDATES = list(fileDATES)
    shortdict = fileDATES[offset:offset+280]
    j=0
    for line in shortdict:
        times_seconds.append(j*3600)
        prices.append(line['Price'])
        DNIs.append(line['DNI'])
        IAMs.append(line['IAM'])
        j = j + 1

    return times_seconds, prices, DNIs, IAMs #return list of times in seconds and prices

def importDNI(filenameDNI, filenameIAM, offset):
    DNI = []
    IAM = []
    #create data frame from csv with pandas module
    df=pd.read_csv(filenameDNI+'.csv', sep=',',skiprows=2) #or:, infer_datetime_format=True)
    dfIAM=pd.read_csv(filenameIAM+'.csv', sep=',') #or:, infer_datetime_format=True)
    
    date=pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    delta=(pd.to_datetime(df['Year'].astype(str)+'-01-01')-date)
    time_data=delta.dt.total_seconds().abs()
    times_seconds_DNI = np.array(time_data)

    times_IAM = np.array(dfIAM['Hours since 00:00 Jan 1'])
    times_IAM_seconds = times_IAM*3600
    times_IAM_seconds = np.add(times_IAM_seconds, -1800)

    DNI = np.array(df['DNI'])
    IAM = np.array(dfIAM['Hourly Data: Field collector incidence angle modifier'])

    return times_seconds_DNI,times_IAM_seconds, DNI, IAM #return list of times in seconds and prices


if __name__ == '__main__':
    t = 30000
    price_schedule,prices = importdict2('/home/rigbac/Projects/ALPACA/HPC_runs/Input/test1_', 50, 0) #start the function with the name of the file
    print(price_schedule[0:20])
    print(prices[0:20])

    time_dni, time_IAM, dni, IAM = importDNI('/home/rigbac/Projects/ALPACA/HPC_runs/Input/TestSyn10', '/home/rigbac/Projects/ALPACA/HPC_runs/Input/IncidenceAngleModifier', 50)
    print(time_dni[0:20])
    print(time_IAM[0:20])
    print(dni[0:20])
    print(IAM[0:20])

    """
    print(price_schedule[10:20],prices[10:20])
    DNI_schedule,DNI = importDNI('/home/rigbac/Projects/ALPACA/HPC_runs/Input/TestSyn10','DNI',0) #start the function with the name of the file

    print(DNI_schedule[10:20],DNI[10:20])
    for i in range(len(price_schedule)):
        if price_schedule[i] <= t < price_schedule[i+1]:
            reward_price = prices[i]
            print(reward_price)

    for i in range(len(DNI_schedule)):
        if DNI_schedule[i] <= t < DNI_schedule[i+1]:
            reward_DNI = DNI[i]
            print(reward_DNI)
    """

    