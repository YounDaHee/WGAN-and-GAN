import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from datetime import datetime 
import os
from PIL import Image
import matplotlib.pyplot as plt 
from tqdm import tqdm
from pytorch_fid import fid_score

# FID 점수를 측정하고자 하는 model_num과 해당 모델을 학습시켰을 때의 noise_div 입력
noise_div = 1
model_num = 230

model_file = f'samples{model_num}/G_w{model_num}.pth.tar'
n_noise = 100
data_folder = '20&30_female'

org_channel = 3 

#FID를 위한 setting
image_num = 667
noise = torch.randn(image_num, n_noise) * noise_div

class Generator(nn.Module):
    # input_size : 노이즈 벡터의 크기
    # tansor_volum : 재구성한 텐서의 높이/너비
    def __init__(self, input_size=n_noise):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 4*4*1024),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            # input: 4 by 4, output: 8 by 8
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # input: 8 by 8, output: 16 by 16
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # input: 16 by 16, output: 32 by 23
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # input: 32 by 32, output: 64 by 64
            nn.ConvTranspose2d(128, org_channel, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        y_ = self.fc(x) 
        y_ = y_.view(y_.size(0), 1024, 4, 4)
        y_ = self.conv(y_) # 업 샘플링
        return y_


def fid():
    G = Generator()
    checkPoint = torch.load(model_file, map_location=torch.device('cpu'))

    G.load_state_dict(checkPoint['state_dict'])

    G.eval()

    directory_path = f"samples{model_num}/generated_images"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with torch.no_grad():  # 그래디언트 계산을 비활성화하여 메모리 사용 줄이기
        for i in tqdm(range(image_num)):
            single_noise = noise[i].unsqueeze(0)
            generated_image = G(single_noise)
            save_image(generated_image, f'{directory_path}/image_{i}.png', normalize=True)

    # FID 점수 계산
    fid_value = fid_score.calculate_fid_given_paths([directory_path, data_folder], batch_size=64, device='cpu', dims=2048)
    print('FID:', fid_value)

    with open(f'samples{model_num}/readme.txt', 'a') as File:
        File.write(f'\nfid score : {fid_value}')

if __name__ == '__main__':
    fid()