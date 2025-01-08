import csv
import matplotlib.pyplot as plt
import seaborn as sns


def read_csv_and_create_lists(path, cols):


    col_list_ = []

    file_name = path + 'output_year_stochastic.csv'

    with open(file_name, mode='r') as file:
        reader = csv.reader(file)

        # Skip the header row if it exists
        #next(reader)
        max_val = 0
        min_val = 0

        for row in reader:
            try:
                value = float(row[cols])  # Column 4 (index 3)
                if value > max_val:
                    max_val = value
                if value < min_val:
                    min_val = value
            except:
                pass

        print(f"max = {max_val}")
        print(f"min = {min_val}")

    with open(file_name, mode='r') as file:
        reader = csv.reader(file)
        t = 0
        for row in reader:
            try:
                time = t
                print(row)
                #if len(row) < 4:
                #    continue  # Skip rows with insufficient columns

                #demand_value = (float(row[0])*10)+53.35  # Column 2 (index 1)
                #price_value = float(row[4])  # Column 3 (index 2)
                #price_value = (float(row[2])*10)+168  # Column 3 (index 2)
                value = float(row[col])  # Column 4 (index 3)

                # Determine the segment based on column 3 value
                segment_index = int(((value-min_val)/(max_val-min_val)) // (1 / 0.99))  # Assuming the value in column 3 ranges from 0 to 100

                print(value,segment_index)
                # Ensure segment index is within range
                if segment_index >= 6:
                    segment_index = 5  # Cap to the last segment if col3_value is at max

                # Append values to the appropriate segment list
                segment_lists[segment_index].append([time,value])
                #segment_lists[segment_index].append(demand_value)
                t+=1
            except:
                pass
    return segment_lists

# Example usage
if __name__ == "__main__":
    csv_file_path = "/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/FinalPPO3/"  # Replace with your CSV file path
    
    lists = read_csv_and_create_lists(csv_file_path, 1)
    var = 'Dispatch_Power'

    for i, segment in enumerate(lists):
        
        print(f"Segment {i + 1}: {segment}")
        try:
            x, y = zip(*segment)

            print(x,y)

            plt.plot(list(x),list(y))
            plt.ylabel(var)
            plt.xlabel('Time / hours')
            #plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/FinalPPO3/Correlation{i}.png")
            #plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/FinalPPO3/"+var+"_Time_year.png")
            #plt.close()
        #plt.plot(segment,lists_y[i-1])
        #plt.show()
        except:
            continue

    lists = read_csv_and_create_lists(csv_file_path, 3)
    var = 'Concrete_Temperature'

    for i, segment in enumerate(lists):
        
        print(f"Segment {i + 1}: {segment}")
        try:
            x, y = zip(*segment)

            print(x,y)

            plt.plot(list(x),list(y))
            #plt.ylabel(var)
            #plt.xlabel('Time / hours')
            #plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/FinalPPO3/Correlation{i}.png")
            plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/FinalPPO3/"+var+"_Time_year.png")
            plt.close()
        #plt.plot(segment,lists_y[i-1])
        #plt.show()
        except:
            continue

    lists = read_csv_and_create_lists(csv_file_path, 8)
    var = 'DNI'

    for i, segment in enumerate(lists):
        
        print(f"Segment {i + 1}: {segment}")
        try:
            x, y = zip(*segment)

            print(x,y)

            plt.plot(list(x),list(y))
            #plt.ylabel(var)
            #plt.xlabel('Time / hours')
            #plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/FinalPPO3/Correlation{i}.png")
            plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/FinalPPO3/"+var+"_Time_year.png")
            plt.close()
        #plt.plot(segment,lists_y[i-1])
        #plt.show()
        except:
            continue

    DNIs = []
    conc_temps = []
    power_demands = []
    times = []