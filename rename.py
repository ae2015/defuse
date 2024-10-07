import os

# Define the directory containing the files
directory = 'data/raw/News1k2024/sport'

# List all files in the directory
files = os.listdir(directory)

# Iterate over the files
for filename in files:
    if filename.startswith('sport_') and filename.endswith('.json'):
        # Extract the number from the filename
        number = int(filename.split('_')[1].split('.')[0])
        
        # Skip sport_1.json
        if number == 1:
            continue
        
        # Decrease the number by one
        new_number = number - 1
        
        # Create the new filename
        new_filename = f'sport_{new_number}.json'
        
        # Get the full paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)

print("Files renamed successfully.")