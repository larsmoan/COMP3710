
import gdown
import zipfile
import os

# Replace 'YOUR_FILE_ID' with the actual file ID from Google Drive
file_id = '16dS_-kOPkBjNgfVViU9VVnDX9h0ZLfHl'

# URL to download the file



# Destination file path where you want to save the downloaded file
output = 'test.zip'

# Download the file
gdown.download(url, output, quiet=False)

# Unzip the downloaded file
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('path_to_extracted_folder')
os.remove(output)


def get_dataset(folder_name: str, file_id: str='kOPkBjNgfVViU9VVnDX9h0ZLfHl'):
    #Check if the folder name exists
    if not os.path.exists(folder_name):
        output = 'dataset.zip'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(get_data_dir())
        os.remove(output)
    else:
        print('Dataset already exists')
        
