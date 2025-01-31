import pandas as pd
import matplotlib.pyplot as plt 

def extract_columns_to_lists(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    df_price = pd.read_csv('/home/rigbac/Projects/ALPACA/HPC_runs/Input/Test2022.csv')

    # Create a dictionary to hold lists for each column
    columns_dict = {}
    vals_dict = {}

    # Extract each column into a list with the column index as the key
    for index, column in enumerate(df.columns):
        print(index)
        if index ==1:
            array = []
            columns_dict[f'column_{index}'] = df[column].tolist()
            for val in columns_dict[f'column_{index}']:
                array.append((val)/(24))
            vals_dict[f'column_{index}'] = array
        if index ==6:
            array = []
            columns_dict[f'column_{index}'] = df[column].tolist()
            for val in columns_dict[f'column_{index}']:
                array.append((val)+48.6)
            vals_dict[f'column_{index}'] = array
        if index ==10:
            array = []
            columns_dict[f'column_{index}'] = df[column].tolist()
            for val in columns_dict[f'column_{index}']:
                array.append(val/100)
            vals_dict[f'column_{index}'] = array
        if index ==8:
            array = []
            columns_dict[f'column_{index}'] = df_price['DNI'].tolist()
            for val in columns_dict[f'column_{index}']:
                array.append((val))
            vals_dict[f'column_{index}'] = array
        if index ==5:
            array = []
            print('here')
            columns_dict[f'column_{index}'] = df_price['Price'].tolist()
            print(columns_dict)
            for i in range(len(columns_dict[f'column_{index}'])):
                if i <  len(df.index):
                    array.append((columns_dict[f'column_{index}'][i]))
                else:
                    continue
            vals_dict[f'column_{index}'] = array
        if index ==9:
            array = []
            times = []
            columns_dict[f'column_{index}'] = df[column].tolist()
            for val in columns_dict[f'column_{index}']:
                array.append((val))
            for i in range(len(columns_dict[f'column_{index}'])):
                val = (columns_dict[f'column_{index}'][i])
                if val == 1.0:
                    #print((vals_dict['column_0']))
                    times.append((vals_dict['column_1'][i]))
            vals_dict[f'column_{index}'] = array
            vals_dict[f'times'] = times

        else:
            continue
            #columns_dict[f'column_{index}'] = df[column].tolist()

    return columns_dict, vals_dict

# Example usage
file_path = '/home/rigbac/Projects/HERON/tests/workshop/ies/IES_o/dispatch_print.csv'
columns_lists,vals_dict = extract_columns_to_lists(file_path)

profit = 0
days = 1/24
for h in range(len(vals_dict[f'column_5'])):
    days+=1/24
    profit += (vals_dict['column_5'][h]*vals_dict['column_6'][h])

print(profit)
ppd = profit/days

errors = 0
for val in vals_dict['column_9']:
    if val == 1.0:
        errors+=1


#print(vals_dict['times'])

# Print the lists
#for key, value in columns_lists.items():
    #print(f"{key}: {value}")

fig, ax1 = plt.subplots(3,1,sharex='all', figsize = (10,10))

fig.suptitle('Full Horizon (HERON) Load Follow \n \n'f'Profit = {profit:.4} USD' +f'  Profit per day = {ppd:.4} USD \n Errors = {errors}',fontsize=16, fontweight='bold')

#ax1[0].set_xlim(0,30)

ax1[0].set_xlabel('time (s)')
ax1[0].set_ylabel('Power Production / MW')
ax1[0].plot(vals_dict['column_1'],vals_dict['column_6'], alpha=0.6, label = 'Power')
ax1[0].tick_params(axis='y')
#for time in vals_dict[f'times']:
    #print(time)
#    ax1[0].axvspan(time, time+(1/24), color='orange', alpha=0.2)


ax1[0].axvspan(0,0, color='orange', alpha=0.2, label = 'Outside Bounds')

ax2 = ax1[0].twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Concrete Temperature / degC')  # we already handled the x-label with ax1
ax2.plot(vals_dict['column_1'],vals_dict['column_10'], alpha = 1,color=color, label='Concrete Temperature')
ax2.tick_params(axis='y')

#color = 'tab:red'
ax1[1].set_xlabel('time (s)')
ax1[1].set_ylabel('DNI / W/m^2')
ax1[1].plot(vals_dict['column_1'],vals_dict['column_8'])
ax1[1].tick_params(axis='y')

#color = 'tab:red'
ax1[2].set_xlabel('time (s)')
ax1[2].set_ylabel('Realtive Price / $/MWh')
ax1[2].plot(vals_dict['column_1'],vals_dict['column_5'])
ax1[2].tick_params(axis='y')

fig.tight_layout(pad = 1.2,h_pad=2)  # otherwise the right y-label is slightly clipped
ax1[0].legend(loc='lower left')
ax2.legend(loc='upper left')
plt.show()

#plt.plot(vals_dict['column_0'],vals_dict['column_3'])
plt.savefig('/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestYears/HERON/DispatchPlot.png')