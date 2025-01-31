import csv
import matplotlib.pyplot as plt
import seaborn as sns


def read_csv_and_create_lists(path):
    # Initialize six lists to hold the values
    segment_lists = [[] for _ in range(6)]
    #segment_lists_y = [[] for _ in range(6)]

    file_name = path + 'output_year_agent.csv'

    with open(file_name, mode='r') as file:
        reader = csv.reader(file)

        # Skip the header row if it exists
        #next(reader)
        max_val = 0
        min_val = 0

        for row in reader:
            temp_value = (float(row[2])*10)+168  # Column 4 (index 3)
            if temp_value > max_val:
                max_val = temp_value
            if temp_value < min_val:
                min_val = temp_value

        print(f"max = {max_val}")
        print(f"min = {min_val}")

    with open(file_name, mode='r') as file:
        reader = csv.reader(file)

        for row in reader:
            print(row)
            #if len(row) < 4:
            #    continue  # Skip rows with insufficient columns

            demand_value = (float(row[1])*10)+53.35  # Column 2 (index 1)
            price_value = float(row[4])  # Column 3 (index 2)
            temp_value = (float(row[3])*10)+168  # Column 3 (index 2)
            #temp_value = float(row[2])  # Column 4 (index 3)

            # Determine the segment based on column 3 value
            segment_index = int(((temp_value-min_val)/(max_val-min_val)) // (1 / 0.99))  # Assuming the value in column 3 ranges from 0 to 100

            print(temp_value,segment_index,price_value,demand_value)
            # Ensure segment index is within range
            if segment_index >= 6:
                segment_index = 5  # Cap to the last segment if col3_value is at max

            # Append values to the appropriate segment list
            segment_lists[segment_index].append([temp_value,demand_value])
            #segment_lists[segment_index].append(demand_value)

    return segment_lists, ma

# Example usage
if __name__ == "__main__":
    csv_file_path = "/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestYears/Agent/"  # Replace with your CSV file path
    
    lists = read_csv_and_create_lists(csv_file_path)

    for i, segment in enumerate(lists):
        
        print(f"Segment {i + 1}: {segment}")
        try:
            x, y = zip(*segment)

            print(x,y)

            sns.regplot(x = list(x), y = list(y), ci=100)
            plt.xlabel('Price / $/MWh')
            plt.ylabel('Power Demand / MW')
            #plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/FinalPPO3/Correlation{i}.png")
            plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestYears/Agent/Correlation_Price.png")
            plt.close()
        #plt.plot(segment,lists_y[i-1])
        #plt.show()
        except:
            continue