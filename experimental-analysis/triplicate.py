import csv

# function to triplicate each row in a csv file and convert to lowercase
def triplicate_csv(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # write the header
        header = next(reader)
        writer.writerow([column.lower() for column in header])
        
        # triplicate each row and convert to lowercase
        for row in reader:
            lowercase_row = [element.lower() for element in row]
            for _ in range(3):  # repeat each row 3 times
                writer.writerow(lowercase_row)

# example usage
input_file = 'participants_info.csv'  # input csv file
output_file = 'participants_info_triplicated.csv'  # output csv file with triplicated rows

triplicate_csv(input_file, output_file)
