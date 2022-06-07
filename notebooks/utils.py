import os
import shutil
import subprocess
import pandas as pd
import random
from skimage import io

import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "There is no such Key"


def get_MedNIST_dataframe(percentage_to_treat=None):
    # this function creates a dataframe with the path of files and GT values
    # for the classification algorithm with pytorch

    base_path = '/content/drive/MyDrive/DL_tutorials/notebooks/'
    if not os.path.exists('MedNIST_0.5.zip'):
        subprocess.run(['curl', '-L', 'https://www.dropbox.com/s/wrbfk4o63f3cn5k/MedNIST_0.5.zip?dl=1',' > ',
                       base_path + 'MedNIST_0.5.zip'])

        shutil.unpack_archive(base_path + 'MedNIST_0.5.zip', base_path)
        shutil.rmtree(base_path + '__MACOSX')

    if percentage_to_treat is None:
        percentage_to_treat = [0.2, 0.2, 0.2, 1., 1., 1.]

    data_path = base_path + 'MedNIST_0.5/'
    list_of_dirs = []
    mp = {}
    i = 0
    for name in os.listdir(data_path):
        if name != 'README.md' and not name.startswith("."):
            list_of_dirs.append(name)
            mp[name] = i
            i += 1
    number_of_dirs = len(list_of_dirs)
    if percentage_to_treat is None:
        percentage_to_treat = [1.] * number_of_dirs

    df_new = pd.DataFrame()

    for i, name in enumerate(list_of_dirs):
        current_dir = data_path + name
        number_of_files = len(os.listdir(current_dir))
        number_of_files_treat = int(percentage_to_treat[i] * number_of_files)

        list_copied_train_files = []

        for j, number in enumerate(random.sample(range(number_of_files), number_of_files_treat)):
            file = os.listdir(current_dir)[number]
            list_copied_train_files.append([name + '/' + file, i, name])

        df_new = pd.concat([df_new, pd.DataFrame(list_copied_train_files)])

    return df_new, mp


class MedicalMNIST(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label, self.annotations.iloc[index, 0]


def create_train_test_dataset(df, train_ratio, batch_size):

    num_total = len(df)
    train_n_files = int(train_ratio * num_total)
    test_n_files = num_total - train_n_files

    base_path = '/content/drive/MyDrive/DL_tutorials/notebooks/'
    dataset = MedicalMNIST(df=df, root_dir=(base_path + 'MedNIST_0.5'),
                           transform=transforms.ToTensor())

    train_set, test_set = torch.utils.data.random_split(dataset, [train_n_files, test_n_files])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate_loss(model, loader, device):
    correct = 0
    total = 0
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch, (x, y, _) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            loss = criterion(scores, y)
            total += loss.item()
    total /= batch + 1
    return total


def evaluate_score(model, loader, device):
    correct = 0
    total = 0
    model.eval()
    list_of_incorrect_preds = []
    with torch.no_grad():
        for x, y, ima in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, pred = scores.max(1)
            for i, check in enumerate(pred != y):
                if check:
                    list_of_incorrect_preds.append([ima[i], y[i].item(), pred[i].item()])
            correct += (pred == y).sum()
            total += pred.size(0)
        print("Accuracy:", correct / total * 100, "%")
        return list_of_incorrect_preds
