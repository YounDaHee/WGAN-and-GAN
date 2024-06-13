
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

from matplotlib.pyplot import  imsave
#%matplotlib inline

MODEL_NAME = 'W-GAN'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 해당 모델 돌리는데에 있어 특이사항 기입
Memo = f""
# 몇번째 버전인지 표기(안바꾸면 다른 파일 덮어 쓰니 주의)
file_num = 108

# 코드 수정을 위한 variable setting
image_size = 64 
num_sample = 5 

# 수정해볼 값들
batch_size = 64
n_noise = 50 # 논문에 언급 X, 다양한 노이즈를 이용해서 다양한 이미지 생성 가능(latent dimension)
noise_div = 0.8 # 노이즈 표준 편차. 논문에서는 0.1 
n_critic = 10 # 논문에서는 5
alpha = 0.00007 # learning rate 논문에서는 0.00005
c = 0.005 # 논문에서는 0.01
max_epoch = 1600

org_channel = 3 # 컬러 = 3, 흑백 = 1

# 학습 시킬 이미지 위치 폴더
data_folder = '20&30_female'

#FID를 위한 setting
image_num = 667
noise = torch.randn(image_num, n_noise) * noise_div

# 생성한 이미지를 저장
directory_path = f"samples{file_num}"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# 실행 시작 시간 메모장에 작성
with open(f'{directory_path}/readme.txt', 'w') as File:
    File.write(f'start\t{datetime.now()}\n\n{Memo}\n\nhyperparameter\nbatch size : {batch_size}\nn_noise : {n_noise}\nnoise_div : {noise_div}\nn_cretic = {n_critic}\nlerning_rate : {alpha}\nclipping : {c}\nmax_epoch : {max_epoch}\n\n')

# 1000step마다 현재 학습 상태를 기준으로 이미지 생성
def get_sample_image(G, n_noise):
    """
        save sample 100 images
    """
    img = np.zeros([image_size*num_sample, image_size*num_sample, org_channel])
    for j in range(num_sample):
        z = torch.randn(num_sample, n_noise).to(DEVICE)*noise_div
        y_hat = G(z).permute(0, 2, 3, 1)
        result = y_hat.cpu().data.numpy()
        img[j*image_size:(j+1)*image_size, :] = np.concatenate([x for x in result], axis=-2)
    img = (img+1)/2
    return img

# 판별자 클래스 정의
# Convolution을 통해 이미지 특징 추출
class Discriminator(nn.Module):
    def __init__(self, in_channel=org_channel, input_size=image_size**2):
        super(Discriminator, self).__init__()

        #신경망 모듈 정의
        self.transform = nn.Sequential(
            # 특징 벡터(각 픽셀마다의 확률)
            nn.Linear(input_size*in_channel, (image_size**2)*in_channel),
            nn.LeakyReLU(0.1),
        )

        #합성곱 신경망
        self.conv = nn.Sequential(
            # 이미지가 커진 만큼 합성곱을 늘려야 된다?
            # 2배씩 감소(ex) image_size = 100, 100 -> 50 -> 25 -> 12 -> ...)
            # 모든 convoltion을 마쳤다면 이후 4x4 윈도우로 평균값 연산

            # 64-> 32
            # 컨볼루션
            nn.Conv2d(in_channel, 1024, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            
            # 32 -> 16
            nn.Conv2d(1024, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # 16 -> 8
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 8 -> 4
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4),
        )
        self.fc = nn.Sequential(
            # channel = 64 / convoution 연산의 결과
            nn.Linear(128, in_channel),
            # 원래 channel 크기인 in_channel로 변환
        )

    def forward(self, x):
        v = x.view(x.size(0), -1)
        y_ = self.transform(v) 
        y_ = y_.view(y_.shape[0], org_channel, image_size, image_size)
        y_ = self.conv(y_)
        y_ = y_.view(y_.size(0), -1)
        y_ = self.fc(y_)
        return y_
    
# 생성자 클래스 정의
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
            # 역 컨볼루션
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

# 생성자/판별자의 가중치를 초기화
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


# 이미지 데이터셋을 학습에 사용할 수 있는 형태로 변경
class MakeDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_list = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_folder, self.image_list[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        # 이미지에 대한 레이블 추가 ("female"을 1로 표시)
        label = 1  # "female"을 나타내는 레이블

        return image, label

# 데이터 변환 설정
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.5], std=[0.5])  # 이미지를 정규화
])

# 이미지를 학습 가능한 DATASET으로 변경
dataset = MakeDataset(data_folder, transform=transform)

data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)


D = Discriminator().to(DEVICE)
G = Generator().to(DEVICE)
G.apply(weights_init)
D.apply(weights_init)
D_opt = torch.optim.RMSprop(D.parameters(), lr=alpha)
G_opt = torch.optim.RMSprop(G.parameters(), lr=alpha)
step = 0
g_step = 0

D_labels = torch.ones([batch_size, 1]).to(DEVICE) # Discriminator Label to real
D_fakes = torch.zeros([batch_size, 1]).to(DEVICE) # Discriminator Label to fake

# 손실 값을 저장할 리스트 초기화
D_losses = []
G_losses = []

# 학습 수행
for epoch in range(max_epoch):
    for idx, (images, labels) in enumerate(data_loader):

        # Training Discriminator
        x = images.to(DEVICE)
        x_outputs = D(x)

        z = torch.randn(batch_size, n_noise).to(DEVICE)*noise_div
        z_outputs = D(G(z))
        D_x_loss = torch.mean(x_outputs)
        D_z_loss = torch.mean(z_outputs)
        D_loss = (D_z_loss - D_x_loss)
        
        D.zero_grad()
        D_loss.backward()
        D_opt.step()
        # Parameter(Weight) Clipping for K-Lipshitz constraint
        for p in D.parameters():
            p.data.clamp_(-c, c)

        if step % n_critic == 0:
            # Training Generator
            z = torch.randn(batch_size, n_noise).to(DEVICE)*noise_div
            z_outputs = D(G(z))
            G_loss = -torch.mean(z_outputs)

            D.zero_grad()
            G.zero_grad()

            G_loss.backward()
            G_opt.step()

            # 현재 step에서의 생성자 모델의 손실값 저장
            G_losses.append(G_loss.item())

        # 판별자 모델의 손실값 저장
        D_losses.append(D_loss.item())

        if step % 500 == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, step, D_loss.item(), G_loss.item()))
        
            G.eval()
            img = get_sample_image(G, n_noise)
            imsave('{}/{}_step{}.jpg'.format(directory_path, MODEL_NAME, str(step).zfill(3)), img)
            G.train()

        if step % 1000 == 0:
            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(G_losses, label="G")
            plt.plot(D_losses, label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f'{directory_path}/loss_plot.png')
            
        step += 1

# 지금까지 결과 저장
def save_checkpoint(state, file_name=f'{directory_path}/checkpoint{file_num}.pth.tar'):
    torch.save(state, file_name)
save_checkpoint({'epoch': epoch + 1, 'state_dict':D.state_dict(), 'optimizer' : D_opt.state_dict()}, f'{directory_path}/D_w{file_num}.pth.tar')
save_checkpoint({'epoch': epoch + 1, 'state_dict':G.state_dict(), 'optimizer' : G_opt.state_dict()}, f'{directory_path}/G_w{file_num}.pth.tar')

# 종료 시간
now = datetime.now()

with open(f'{directory_path}/readme.txt', 'a') as File:
    File.write(f'end\t{datetime.now()}')

