import sys

def combine_files(input_filenames, output_filename):
    combined_lines = []

    with open(output_filename, 'w') as output_file:

        for filename in input_filenames:
            print(filename)
            first = True
            with open(filename, 'r') as file:
                lines = file.readlines()

                # If it's the first file, include all lines except for the last "</LesHouchesEvents>" line
                if not combined_lines or first:
                    last_index = next((i for i, line in enumerate(lines) if '</LesHouchesEvents>' in line), None)
                    if last_index is not None:
                        output_file.writelines(lines[:last_index])
                        first=False
                else:
                    # Find the index of the first occurrence of "<event>" and the occurance of "</LesHouchesEvents>"
                    first_index = next((i for i, line in enumerate(lines) if '<event>' in line), None)
                    last_index = next((i for i, line in enumerate(lines) if '</LesHouchesEvents>' in line), None)

                    # Include lines after the first occurrence of "<event>"
                    if first_index is not None and last_index is not None:
                        #combined_lines.extend(lines[first_index:last_index])
                        output_file.writelines(lines[first_index:last_index])

        # add last line:
        #combined_lines.append('</LesHouchesEvents>')
        output_file.writelines('</LesHouchesEvents>')
        ## Write the combined lines to the output file
        #with open(output_filename, 'w') as output_file:
        #output_file.writelines(combined_lines)

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) < 4:
        print("Usage: python script.py output_filename input_file1 input_file2 ...")
        sys.exit(1)

    # Extract command-line arguments
    output_file = sys.argv[1]
    input_files = sys.argv[2:]

    # Call the combine_files function
    combine_files(input_files, output_file)

