# Weather image classification using convolutional neural network.
# @author Chenjie Wu

# Using Adam (default lr, beta, etc.):
# Best cross_entropy loss: 0.3 - 0.35
# Best accuracy (correct_predictions / all_predictions): 0.9-0.92
# Gotten in: 15-25 epoches

# Using SGD (Hyp. params listed below)
# with 3-stage loss_related LR adjustment:
# Best cross_entropy loss: 0.35-0.5
# Best accuracy: 0.88
# Gotten in: 100+ epoches.

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import numpy as np
import torch.cuda as cuda
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

# ---------------------------Hyperparameters------------------------#

LEARNING_RATE = 0.05
BATCH_SIZE = 20
EPOCH = 200

# --------------------------END Hyperparameters---------------------#


ROOT_DIR = "./data/weather_dataset"


# randomly generate 2 lists of indices: (train_ids, val_ids)
def split_indices(n, val_pct):
    indices = np.random.permutation(n)
    bound = int(n * val_pct)
    return indices[bound:], indices[:bound]


def get_default_device():
    if cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')  # incase cuda is unavaliable.


# move data and model to device. This operation may take while
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]  # breakdown element and
        # recursively put elements into device if the data is list or tuple.

    return data.to(device, non_blocking=True)  # This may return a pointer pointing to the data in GPU


# A wrapped dataloader that put data into gpu in BATCH-WISE fashion.
# This can save GPU dram.
class DeviceDataloader():
    # Constructor
    # @param data_loader A DataLoader instance that need to be operated.
    # @param device The device going to put on.
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    # iterator function.
    def __iter__(self):
        for batch_tuple in self.data_loader:
            yield to_device(batch_tuple, self.device)  # do to_device. Yield is non-stop return,

    # length of self.data_loader
    def __len__(self):
        return len(self.data_loader)


class WR_Conv_Model(nn.Module):

    def accuracy(self, prediction, target):
        _, max_pos = torch.max(prediction, dim=1)
        return torch.tensor(torch.sum(max_pos == target).item() / len(max_pos))

    def __init__(self):
        super(WR_Conv_Model, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 6, kernel_size=3, padding=1, stride=1)
        # now BSx6x128x128
        self.conv_layer2 = nn.Conv2d(6, 12, kernel_size=3, padding=1, stride=1)
        # now BSx12x64x64
        self.conv_layer3 = nn.Conv2d(12, 24, kernel_size=3, padding=1, stride=1)
        # now BSx24x32x32
        self.conv_layer4 = nn.Conv2d(24, 48, kernel_size=3, padding=1, stride=1)
        # now BSx48x16x16
        self.conv_layer5 = nn.Conv2d(48, 96, kernel_size=3, padding=1, stride=1)
        # now BSx96x8x8
        self.conv_layer6 = nn.Conv2d(96, 192, kernel_size=3, padding=1, stride=1)
        # now BSx192x4x4

        # here data is flatten from dim=1
        self.linear_layer1 = nn.Linear(192 * 4 * 4, 256)
        # now BSx256
        self.linear_layer2 = nn.Linear(256, 4)
        # now prediction is getted: BSx4

    def forward(self, input_batch):
        out = self.conv_layer1(input_batch)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv_layer2(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv_layer3(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv_layer4(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv_layer5(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv_layer6(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = out.view(input_batch.size(0), -1)

        out = self.linear_layer1(out)
        out = nn.functional.relu(out)

        out = self.linear_layer2(out)

        return out

    def training_step(self, data_batch, target_batch):
        prediction = self(data_batch)
        loss = nn.functional.cross_entropy(prediction, target_batch)
        return loss

    def validation_step(self, data_batch, target_batch):
        prediction = self(data_batch)
        accuracy = self.accuracy(prediction, target_batch)
        loss = nn.functional.cross_entropy(prediction, target_batch)
        return {"validation loss": loss, "validation accuracy": accuracy}


# ---------------------------Prepare and load the data----------------------#
# open dataset
dataset = ImageFolder(ROOT_DIR + "/train", transform=ToTensor())
print("Dataset length: " + str(len(dataset)))
# split the indices into train and validation
train_ids, val_ids = split_indices(len(dataset), 0.1)  # 10% validation set
# transform data into dataloader
train_loader = DataLoader(dataset, BATCH_SIZE, sampler=SubsetRandomSampler(train_ids))
val_loader = DataLoader(dataset, BATCH_SIZE, sampler=SubsetRandomSampler(val_ids))

train_loader = DeviceDataloader(train_loader, get_default_device())
val_loader = DeviceDataloader(val_loader, get_default_device())
print("--------------Data transformed to: " + str(get_default_device()) + "--------------------")

# ----------------------------------END Prepare---------------------------------------#

# --------------------------------Construct model and optimizer-----------------------#
model = WR_Conv_Model()
to_device(model, get_default_device())

# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# ---------------------------------END Construct---------------------------------------#

# --------------------------------Train------------------------------------------#
print("Here")
'''stage_1 = False
stage_2 = False
stage_3 = False'''

for i in range(EPOCH):
    # time0 = time.time()
    for data_batch, target_batch in train_loader:
        # print("Sample prediction: " + str(nn.functional.softmax(prediction, dim=1)[0]))
        loss = model.training_step(data_batch, target_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # time1 = time.time()
    print("EPOCH: " + str(i))
    # print("Time cost in current epoch: " + str(time1 - time0))
    if i % 5 == 0 or i == EPOCH - 1:
        validation_data = [model.validation_step(validation_batch, validation_targets) for
                           validation_batch, validation_targets in val_loader]
        losses = [data['validation loss'].data for data in validation_data]
        avg_loss = sum(losses) / len(losses)
        accs = [data['validation accuracy'].data for data in validation_data]
        avg_accuracy = sum(accs) / len(accs)
        print("Average loss: " + str(avg_loss))
        print("Average accuracy: " + str(avg_accuracy))
        '''if avg_loss < 0.5 and stage_1 == False:
            for param in optimizer.param_groups:
                param['lr'] /= 5
                stage_1 = True
                print("learning rate decreased to stage 1!")

        if avg_loss < 0.4 and stage_2 == False:
            for param in optimizer.param_groups:
                param['lr'] /= 5
                stage_2 = True
                print("learning rate decreased to stage 2!")

        if avg_loss < 0.35 and stage_3 == False:
            for param in optimizer.param_groups:
                param['lr'] /= 5
                stage_3 = True
                print("learning rate decreased to stage3!")'''
