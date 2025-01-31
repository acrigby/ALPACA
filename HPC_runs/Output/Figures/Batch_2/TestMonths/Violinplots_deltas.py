import pandas as pd
import matplotlib.pyplot as plt 
import scipy.stats as stats
import numpy as np

def extract_columns_to_lists(agent_type,filename, month):

    file_path = '/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestMonths/' + agent_type + '/Month_' + str(month) + '/' + filename
    # Read the CSV file
    df = pd.read_csv(file_path, nrows = 720)

    df_price = pd.read_csv('/home/rigbac/Projects/ALPACA/HPC_runs/Input/Test2022.csv')

    start_offset = month*720

    # Create a dictionary to hold lists for each column
    columns_dict = {}
    vals_dict = {}

   # Extract each column into a list with the column index as the key
    for index, column in enumerate(df.columns):
        #print(index)
        if index ==0:
            array = []
            columns_dict[f'column_{index}'] = df[column].tolist()
            for val in columns_dict[f'column_{index}']:
                array.append((val-1e6)/(3600*24))
            vals_dict[f'times'] = array
        if index ==1:
            array = []
            columns_dict[f'column_{index}'] = df[column].tolist()
            for val in columns_dict[f'column_{index}']:
                array.append((val*10)+4.35)
            vals_dict[f'power'] = array
        if index ==2:
            array = []
            columns_dict[f'column_{index}'] = df[column].tolist()
            for val in columns_dict[f'column_{index}']:
                array.append(((val)*202000)+202000)
            vals_dict[f'pressure'] = array
        if index ==3:
            array = []
            columns_dict[f'column_{index}'] = df[column].tolist()
            for val in columns_dict[f'column_{index}']:
                array.append((val*10)+168)
            vals_dict[f'conc_temp'] = array
        if index ==5:
            array = []
            #print('here')
            columns_dict[f'column_{index}'] = df_price['Price'].tolist()
            #print(columns_dict)
            for i in range(len(columns_dict[f'column_{index}'])):
                if i <  len(df.index):
                    array.append((columns_dict[f'column_{index}'][i + start_offset]))
                else:
                    continue
            vals_dict[f'price'] = array
        if index ==8:
            array = []
            columns_dict[f'column_{index}'] = df[column].tolist()
            for val in columns_dict[f'column_{index}']:
                array.append((val*2000))
            vals_dict[f'DNI'] = array
        if index ==9:
            array = []
            columns_dict[f'column_{index}'] = df[column].tolist()
            for val in columns_dict[f'column_{index}']:
                array.append((val))
            vals_dict[f'errors'] = array


        else:
            columns_dict[f'column_{index}'] = df[column].tolist()

    times = []
    """
    errors = 0
    for i in range(len(columns_dict[f'column_{index}'])):
        val = (columns_dict[f'column_{index}'][i])
        if ((vals_dict['conc_temp'][i] < 164) or vals_dict['conc_temp'][i] > 172):
            print(vals_dict['conc_temp'][i])
            times.append((vals_dict['times'][i]))
            errors+=1
        elif ((vals_dict['pressure'][i] < 101000) or vals_dict['pressure'][i] > 160000):
            print(vals_dict['pressure'][i])
            times.append((vals_dict['times'][i]))
            errors+=1
    """
    errors = 0
    #print(vals_dict['errors'])
    for i in range(len(vals_dict['errors'])):
        val = vals_dict['errors'][i]
        if val == 1.0:
            times.append((vals_dict['times'][i]))
            errors+=1
    vals_dict[f'times_errors'] = times

    return columns_dict, vals_dict, errors

def plotmonth(vals_dict,errors,agent_type,month):
    profit = 0
    days = 1/24
    for h in range(len(vals_dict[f'price'])):
        days+=1/24
        profit += (vals_dict['price'][h]*vals_dict['power'][h])

    #print(profit)
    ppd = profit/days

    n_profit = 0
    n_days = 1/24
    
    for h in range(len(vals_dict[f'price'])):
        if vals_dict['errors'][h] == 1.0:
            n_days+=1/24
            n_profit += 0
        elif (h>1 and vals_dict['errors'][h-1] == 1.0):
            n_days+=1/24
            n_profit += 0
        else:
            n_days+=1/24
            n_profit += ((vals_dict['price'][h])/(sum(vals_dict[f'price'])/len(vals_dict[f'price'])))*vals_dict['power'][h]
            #n_profit += ((vals_dict['price'][h])/(sum(vals_dict[f'price'])/len(vals_dict[f'price'])))*vals_dict['power'][h]
    
    """
    for h in range(len(vals_dict[f'price'])):
        n_days+=1/24
        #n_profit += ((vals_dict['price'][h]-min(vals_dict[f'price']))/(max(vals_dict[f'price'])-min(vals_dict[f'price'])))*vals_dict['power'][h]
        n_profit += ((vals_dict['price'][h])/(sum(vals_dict[f'price'])/len(vals_dict[f'price'])))*vals_dict['power'][h]
    """
    print(n_days)
    nppd = n_profit/n_days

    """
    errors = 0
    for val in vals_dict['column_9']:
        if val == 1.0:
            errors+=1

    errors = 0
    for k in range(len(vals_dict[f'column_3'])):
        if (vals_dict['column_3'][k] < 164) or vals_dict['column_3'][k] > 172:
            errors+=1
    """
    
    #print(vals_dict['times'])

    # Print the lists
    #for key, value in columns_lists.items():
        #print(f"{key}: {value}")

    fig, ax1 = plt.subplots(3,1,sharex='all', figsize = (10,10))

    fig.suptitle(agent_type + f' Load Follow Month {month} \n \n'f'Profit = {profit:.4} USD' +f'  Profit per day = {ppd:.4} USD \n Errors = {errors}',fontsize=16, fontweight='bold')


    ax1[0].set_xlabel('time (s)')
    ax1[0].set_ylabel('Power Production / MW')
    ax1[0].plot(vals_dict['times'],vals_dict['power'], alpha=0.6, label = 'Power')
    ax1[0].tick_params(axis='y')
    for time in vals_dict[f'times_errors']:
        #print(time)
        ax1[0].axvspan(time, time+(1/24), color='orange', alpha=0.2)


    ax1[0].axvspan(0, 0, color='orange', alpha=0.2, label = 'Outside Bounds')

    ax2 = ax1[0].twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Concrete Temperature / degC')  # we already handled the x-label with ax1
    ax2.plot(vals_dict['times'],vals_dict['conc_temp'], color=color, label='Concrete Temperature')
    ax2.tick_params(axis='y')

    #color = 'tab:red'
    ax1[1].set_xlabel('time (s)')
    ax1[1].set_ylabel('DNI / W/m^2')
    ax1[1].plot(vals_dict['times'],vals_dict['DNI'])
    ax1[1].tick_params(axis='y')

    #color = 'tab:red'
    ax1[2].set_xlabel('time (s)')
    ax1[2].set_ylabel('Realtive Price / $/MWh')
    ax1[2].plot(vals_dict['times'],vals_dict['price'])
    ax1[2].tick_params(axis='y')

    fig.tight_layout(pad = 1.2,h_pad=2)  # otherwise the right y-label is slightly clipped
    ax1[0].legend(loc='lower left')
    ax2.legend(loc='upper left')
    plt.show()

    #plt.plot(vals_dict['column_0'],vals_dict['column_3'])
    #plt.savefig('/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestMonths/' + agent_type + '/Month_' + str(month) + '/DispatchPlot.png')
    return profit, ppd, n_profit, nppd


# Example usage
agent_types = ['SS_New2', 'Functional', 'Agent', 'HERON']
agent_names = ['Low Load Follow', 'High Load Follow', 'DRL Agent', 'MILP Solution']

dict_nppd = {'SS_New2':[[],[]], 'Functional':[[],[]], 'Agent':[[],[]], 'HERON':[[],[]]}

for month in range(0,12,1):
    for agent_type in ['SS_New2', 'Functional', 'Agent', 'HERON']:
        if agent_type == 'SS_New2':
            filename = f'output_month_{month}_SS.csv'
        elif agent_type == 'Functional':
            filename = f'output_month_{month}_functional.csv'
        elif agent_type == 'Agent':
            filename = f'output_month_agent.csv'
        elif agent_type == 'HERON':
            filename = f'output_month_{month}_HERON.csv'

        try:
            columns_lists,vals_dict,errors = extract_columns_to_lists(agent_type,filename, month)
            profit, ppd, monthly_normalised_profit, nppd_month = plotmonth(vals_dict,errors,agent_type,month)
            print('Agent = ' + agent_type + f', Month = {month}, Monthly Profit Nomalised = {monthly_normalised_profit}, Normalised Profit per Day = {nppd_month}, Errors = {errors}')
            dict_nppd[agent_type][0].append(nppd_month)
            dict_nppd[agent_type][1].append(errors)
        except:
            continue
    
print(dict_nppd)

plt.clf()
fig = plt.figure(figsize=(10,7))

dict_list = []
dict_list_delta = [[],[],[],[]]
keys = []

for k, v in dict_nppd.items():
      print(k,v[0])
      dict_list.append(v[0])
      keys.append(k)

for i in range(len(dict_list)):
    for j in range(len(dict_list[0])):
        dict_list_delta[i].append(dict_list[i][j] - dict_list[0][j])

plt.title('Violin Plot of profitability accounting for OOB operation')
plt.ylabel('Normalised Profit per Day Change from Low Load Follow Case / MWh')
plt.xlabel('Agent Type')
plt.violinplot(dict_list_delta, showmeans=True) 
plt.xticks([y + 1 for y in range(len(dict_list))],labels=agent_names) 
#plt.show()
plt.savefig('/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestMonths/Violinplot_profit_delta_subhours.png')

print('Mean of Low Load Follow is = ' + str(sum(dict_list_delta[0])/12))
print('Mean of High Load Follow is = ' + str(sum(dict_list_delta[1])/12))
print('Mean of Agent Load Follow is = ' + str(sum(dict_list_delta[2])/12))
print('Mean of MILP Load Follow is = ' + str(sum(dict_list_delta[3])/12))

plt.clf()
fig = plt.figure(figsize=(10,7))

Agent_arr = np.array(dict_list_delta[2])
Heron_arr = np.array(dict_list_delta[3])

print(stats.ttest_ind(a=Agent_arr, b=Heron_arr, equal_var=True))


"""

dict_list = []
keys = []

for k, v in dict_nppd.items():
      print(k,v[1])
      dict_list.append(v[1])
      keys.append(k)

plt.title('Violin plots of Errors')
plt.ylabel('Hours with bounds crossed')
plt.xlabel('Agent Type')
plt.violinplot(dict_list, showmeans=True)
plt.xticks([y + 1 for y in range(len(dict_list))],labels=agent_names) 
#plt.show()
plt.savefig('/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestMonths/Violinplot_errors_delta.png')

print('Mean of Low Load Follow is = ' + str(sum(dict_list[0])/12))
print('Mean of High Load Follow is = ' + str(sum(dict_list[1])/12))
print('Mean of Agent Load Follow is = ' + str(sum(dict_list[2])/12))
print('Mean of MILP Load Follow is = ' + str(sum(dict_list[3])/11))

"""