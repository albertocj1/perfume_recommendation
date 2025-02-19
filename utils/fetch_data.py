import os
import zipfile
import warnings

# Suppress SyntaxWarnings if any
warnings.filterwarnings("ignore", category=SyntaxWarning)

'''
    To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com
    Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account)
    and select 'Create API Token'. This will download a file named 'kaggle.json' to your computer.
    Place this file in the location: ~/.kaggle/kaggle.json on Linux/Mac,
    or at C:/Users/<Windows-username>/.kaggle/kaggle.json on Windows.
'''

# Kaggle dataset identifier
dataset_slug = "nandini1999/perfume-recommendation-dataset"

# Define the target folder for saving the dataset
target_folder = r"../data/raw"  # Raw string to avoid issues with backslashes
os.makedirs(target_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Download the dataset into the specified folder
print(f"Downloading dataset: {dataset_slug}")
os.system(f'kaggle datasets download -d {dataset_slug} -p "{target_folder}"')

# Extract any ZIP files in the target folder
for file in os.listdir(target_folder):
    if file.endswith(".zip"):
        zip_path = os.path.join(target_folder, file)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_folder)
        print(f"Extracted: {file}")
        os.remove(zip_path)  # Remove the ZIP file after extraction

print(f"Dataset downloaded and extracted to: {os.path.abspath(target_folder)}")