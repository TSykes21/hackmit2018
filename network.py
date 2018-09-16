# Influenced by: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56

import torch
import torch.nn as nn
from scipy.io import wavfile as wav
import numpy as np
from numpy.fft import fft, fftfreq
import argparse
import os
import random
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('train', nargs='?')
args = parser.parse_args()
init_train = False
if args.train: init_train = True


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
num_classes = 7
batch_size = 10
learning_rate = 0.001




def createInput(filepath):
    rate, data = wav.read(filepath)
    try:
        if data.shape[1] > 1:
            data = data.T[0]
    except IndexError:
        pass

    i = 0
    while data[i] == 0:
        i += 1
    data = data[i:i+rate]

    if data.shape[0] < rate:
        zeroes = np.zeros(rate - data.shape[0])
        data = np.concatenate((data,zeroes), axis=0)

    reduced = []
    ct = 0
    len = 0
    for pt in data:
        if ct % 75 == 0:
            reduced.append(pt)
            len += 1
        ct += 1

    data = np.asarray(reduced)
    data_len = len

    # create spectrograph
    spectrograph = np.zeros((20,14))
    time_frame = int(data_len/20)
    column = 0
    for i in range(time_frame, data_len + 1, time_frame):
        freqs = fftfreq(time_frame) # cycles/second, if data_len is in seconds

        fft_out = fft(data[i - time_frame:i])
        true_fft = 2.0 * np.abs(fft_out)/data_len

        mask = freqs > 0
        freqs = freqs[mask]
        true_fft = true_fft[mask]

        # fill row of graph
        for j in range(0, true_fft.size):
            spectrograph[int(i/time_frame)-1,j] += true_fft[j]
            column += 1
        column = 0
    spectrograph = torch.from_numpy(spectrograph).float()
    return spectrograph

def toOneHot(num):
    one_hot = torch.zeros(7)
    one_hot[num] = 1
    return one_hot


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(480, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def evaluate(filepath, modelpath):
    with torch.no_grad():
        model = ConvNet(num_classes)
        model.load_state_dict(torch.load(modelpath))
        batch = torch.Tensor()
        for i in range(batch_size):
            input = createInput(filepath)
            if len(input.shape) == 2:
                input.unsqueeze_(0).unsqueeze_(0)
            elif len(input.shape) == 4:
                input.reshape(input.shape[2], input.shape[3])
            batch = torch.cat((batch, input))

        outputs = model(batch)

        lan_codes = ['ar', 'zh-CN', 'en', 'fr', 'ja', 'ru', 'es']

        outputs = outputs[0].numpy()

        max = 0
        idx = 0
        cnt = 0
        for elem in outputs:
            if elem > max:
                max = elem
                idx = cnt
                print(max,idx)
            cnt += 1

        return lan_codes[idx]



if init_train:
    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training data collection
    data_list = []
    langs = ["Arabic", "Chinese", "English", "French", "Japanese", "Russian", "Spanish"]
    for subdir, dirs, files in os.walk('./data'):
        for file in files:
            if '.wav' not in file: continue
            lang_idx = 0
            for lang in langs:
                if lang in subdir:
                    lang_idx = langs.index(lang)
            data_list.append(((createInput(os.path.join(subdir, file))),lang_idx))
    random.shuffle(data_list)

    total_step = len(data_list)
    for epoch in range(num_epochs):

        batch = torch.Tensor()
        labels = torch.Tensor()
        datalist_len = len(data_list)
        for i in range(int(len(data_list)/batch_size)):
            for j in range(batch_size):
                image = data_list[10 * i + j][0]
                if len(image.shape) == 2:
                    image.unsqueeze_(0).unsqueeze_(0)
                elif len(image.shape) == 4:
                    image.reshape(image.shape[2],image.shape[3])

                batch = torch.cat((batch, image))
                one_hot = toOneHot(data_list[10 * i + j][1])
                label = one_hot.unsqueeze_(0)
                labels = torch.cat((labels, label))


            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}],  Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
            batch = torch.Tensor()
            labels = torch.Tensor()
        random.shuffle(data_list)

    torch.save(model.state_dict(), 'test.model')


else:


    model = ConvNet(num_classes)
    model.load_state_dict(torch.load("test.model"))
    batch = torch.Tensor()
    for i in range(batch_size):
        input = createInput('./data/party.wav')
        if len(input.shape) == 2:
            input.unsqueeze_(0).unsqueeze_(0)
        elif len(input.shape) == 4:
            input.reshape(input.shape[2], input.shape[3])
        batch = torch.cat((batch, input))

    outputs = model(batch)

    spectrograph = batch[0].numpy()
    spectrograph = spectrograph.reshape(spectrograph.shape[1],spectrograph.shape[2])
    plt.imshow(spectrograph, cmap=plt.cm.gray)
    plt.show()





