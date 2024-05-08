import os

# Directory containing the individual text files
directory = "./runs/detect/track3/labels"

# Output file path
output_file = "aggregated_results.txt"

# Dictionary to store the aggregated results
aggregated_results = {}

# Iterate over all the text files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        # Extract the frame number from the filename
        frame_num = filename.split("_")[-1].split(".")[0]
        
        # Open the individual text file
        with open(os.path.join(directory, filename), "r") as infile:
            # Read each line in the text file
            for line in infile:
                # Split the line into its components
                data = line.strip().split()
                
                if len(data) == 6:
                    # Extract the relevant information
                    class_id = data[0]
                    x = float(data[1]) * 1920
                    y = float(data[2]) * 1080
                    w = float(data[3]) * 1920
                    h = float(data[4]) * 1080
                    confidence_score = data[5]
                    
                    if frame_num not in aggregated_results:
                        aggregated_results[frame_num] = []
                    
                    aggregated_results[frame_num].append(f"{class_id} {x:.4f} {y:.4f} {w:.4f} {h:.4f} {confidence_score}")

# Sort the frame numbers numerically
sorted_frame_nums = sorted(aggregated_results.keys(), key=int)

# Open the output file in write mode
with open(output_file, "w") as outfile:
    # Iterate over the sorted frame numbers
    for frame_num in sorted_frame_nums:
        # Write the frame number and its corresponding results to the output file
        for result in aggregated_results[frame_num]:
            outfile.write(f"{frame_num} {result}\n")