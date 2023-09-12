import os
import gdown
import zipfile

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets


def get_dataset(folder_name: str, file_id: str='16dS_-kOPkBjNgfVViU9VVnDX9h0ZLfHl'):
    if not os.path.exists(folder_name):
        output = 'dataset.zip'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(output)
    else:
        print('Dataset is already downloaded')
        
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=folder_name, transform=transform)
    return dataset