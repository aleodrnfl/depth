## 기본 unet에 cross attention 얹기
## 이미지 512 -> 256으로 리사이즈

import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import random
import librosa
import numpy as np
import os
import json
from PIL import Image
from tqdm import tqdm
import csv
import cv2
import wandb  

wandb.init(
    project="model2",  # 🔥 프로젝트 이름
    config={
        "batch_size": 16,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "loss_function": "MSEloss",
        "optimizer": "Adam",
    }
)
num_epochs = 50
best_mse_loss = float("inf")  # UNet 모델의 최적 MSE Loss
early_stop_num = 100
num_ss = 15



""" 🔥 1. wav_to_mel 변환 """
def wav_to_mel(wav_path, output_size=(224, 224)):
    y, sr = librosa.load(wav_path, sr=None)

    stft_result = librosa.stft(y, n_fft=512, win_length=64, hop_length = 16)
    stft_mag, _ = librosa.magphase(stft_result)
    
    mel_resized = cv2.resize(stft_mag, output_size, interpolation=cv2.INTER_CUBIC)
    
    min_val = np.min(mel_resized)
    max_val = np.max(mel_resized)
    if max_val - min_val == 0:
        mel_resized = np.zeros_like(mel_resized)  # 모든 값을 0으로 설정
    else:
        mel_resized = (mel_resized - min_val) / (max_val - min_val)
        
    mel_image = mel_resized[np.newaxis, ...]    # (1, 256, 256)
    return mel_image

""" 🔥 2. 데이터 텐서로 변환 """
class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.resize = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            data_point = self.data_list[idx]
            rgb_path = data_point['rgb']
            depth_path = data_point['depth']
            
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {rgb_path} 또는 {depth_path}")
                
            rgb = np.array(Image.open(rgb_path)) / 255.0
            rgb = rgb.transpose(2, 0, 1)    # (C, H, W)
            rgb = torch.tensor(rgb, dtype=torch.float32)
            rgb = self.resize(rgb)
            
            depth = np.array(Image.open(depth_path).convert('L')) / 255.0
            depth = np.expand_dims(depth, axis=0)   #(1, H, W)
            depth = torch.tensor(depth, dtype=torch.float32)
            depth = self.resize(depth)
            
            sounds = data_point['sound']
            sound_items = []
            for data in sounds:
                # ss = wav_to_mel(data["ss"])  # (1, 512, 512)
                left_ir = wav_to_mel(data["leftIR"])  
                right_ir = wav_to_mel(data["rightIR"])  
                
                sound = np.concatenate([left_ir, right_ir], axis=0) # (2, 512, 512)
                sound = torch.tensor(sound, dtype=torch.float32)    # 2채널 텐서로
                sound = self.resize(sound)
                sound_items.append(sound)

            sound_tensor = torch.stack(sound_items, dim=0)  # (20, 1, 512, 512)

            return rgb, depth, sound_tensor

        except Exception as e:
            print(f"데이터 로드 중 오류 발생 (index: {idx}) - {e}")
            return None  
        
""" 🔥 3. U-Net 모델 정의 """
class SoundEnc(nn.Module):
    def __init__(self):
        super(SoundEnc, self).__init__()
        # Pretrained ResNet-18 모델 불러오기
        resnet18 = models.resnet18(pretrained=True)

        # 입력 채널을 2로 변경 (conv1 수정)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet18.conv1.weight.data[:, :2]  # 기존 가중치 중 앞 2개 채널만 가져옴

        # ResNet-18의 기본 블록을 그대로 사용
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool

        self.layer1 = resnet18.layer1  # (64, 56, 56)
        self.layer2 = resnet18.layer2  # (128, 28, 28)
        self.layer3 = resnet18.layer3  # (256, 14, 14)
        
        # layer4의 출력을 1024 채널로 확장
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )  # (1024, 14, 14)

    def forward(self, x):
        x = self.conv1(x)   # (2, 224, 224) -> (64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # (64, 56, 56)

        x = self.layer1(x)  # (64, 56, 56)
        x = self.layer2(x)  # (128, 28, 28)
        x = self.layer3(x)  # (256, 14, 14)
        x = self.layer4(x)  # (1024, 14, 14)

        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)    
        self.conv_block = ConvBlock(out_channels * 2, out_channels)     

    def forward(self, x, skip): # 예시: in_channels=1024, out_channels=512
        x = self.up(x)  # (batch_size, 512, 2H, 2W)
        x = torch.cat([x, skip], dim=1) # (batch_size, 512 + 512, 2H, 2W)
        return self.conv_block(x) # (batch_size, 512, 2H, 2W)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
    

    def forward(self, query, key, value):
        """
        query: (batch_size, C, H, W)
        key: (batch_size, C, H, W)
        value: (batch_size, C, H, W)
        """
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        Q = Q.flatten(2).transpose(1, 2)  # (B, H*W, C)
        K = K.flatten(2).transpose(1, 2)  # (B, H*W, C)
        V = V.flatten(2).transpose(1, 2)  # (B, H*W, C)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  # Scaled dot-product attention
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H*W, H*W)

        attn_output = torch.matmul(attn_weights, V)  # (B, H*W, C)
        attn_output = attn_output.transpose(1, 2).reshape(query.shape)  # (B, C, H, W)

        return attn_output
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.e1 = EncoderBlock(3, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        
        self.b = ConvBlock(512, 1024)
        self.attn = CrossAttention(1024)
        
        self.sound = SoundEnc()

        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        self.outputs = nn.Conv2d(64, 1, 1) # 커널 사이즈 1

    def forward(self, x, sound_input):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.b(p4)
        
        sound = self.sound(sound_input)
        b = self.attn(b, sound, sound)
        # sound_b = self.attn(sound, b, b)
        # combined = torch.cat([b, sound_b], dim=1)  # (B, C*2, H, W)
        
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        depth_output = self.outputs(d4) #(1, 1, 512, 512)

        return depth_output


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

with open("/home/mihyun/server_data/model/model2/train_data_pano2_semantic.json", "r") as file:
    train_data_zip = json.load(file)
with open("/home/mihyun/server_data/model/model2/val_data_pano2_semantic.json", "r") as file:
    valid_data_zip = json.load(file)

train_dataset = CustomDataset(train_data_zip)
val_dataset = CustomDataset(valid_data_zip)

train_size = int(0.1 * len(train_dataset))  
train_subset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])

val_size = int(0.1 * len(val_dataset))  
val_subset, _ = random_split(val_dataset, [val_size, len(val_dataset) - val_size])

batch = 16
# DataLoader 생성
train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True, num_workers=8)
val_loader = DataLoader(val_subset, batch_size=batch, shuffle=False, num_workers=8)

""" 🔥 5. 모델, 손실, 옵티마이저 정의 """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# """ 모델 중간부터 다시 돌릴 때 """
# train_checkpoint_path = "/home/mihyun/server_data/model/model2/0207_output/train_2_30.pth"
# train_checkpoint = torch.load(train_checkpoint_path, map_location=device)

# model.load_state_dict(train_checkpoint['model_state_dict'])
# optimizer.load_state_dict(train_checkpoint['optimizer_state_dict'])

# start_epoch = train_checkpoint['epoch'] + 1
# print(f"Loaded training checkpoint from {train_checkpoint_path} (epoch {train_checkpoint['epoch']})")

wandb.watch(model, log="all")
criterion_mse = nn.MSELoss()

""" 🔥 6. 학습 루프 """
for epoch in range(start_epoch, num_epochs):
    model.train()   
    mse_losses = 0.0
    
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")): 
        if batch is None:
            continue 
        rgb, depth, sound_tensor = batch # (B, 3, H, W), (B, 1, H, W), (B, 20, 3, H, W)
        sound_losses = []   
        
        for sound_idx in range(sound_tensor.size(1)): 
            one_sound = sound_tensor[:, sound_idx]  # (B, 3, H, W)
            
            # inputs = torch.cat([rgb, one_sound], dim=1)   # (B, 6, H, W)
            inputs, one_sound, targets = rgb.to(device), one_sound.to(device), depth.to(device)
            outputs = model(inputs, one_sound)

            mse_loss = criterion_mse(outputs, targets) 
            sound_losses.append(mse_loss.item())       
            
            optimizer.zero_grad()
            mse_loss.backward()  
            optimizer.step()
        
        #테스트할 때 아래 줄 Json으로 저장하면 됨 (입력값이랑 연관지어서) (loss값 가장 적은 항목도 인덱스 알아내서) 입력값은 배치1 가능 (1, 3, 512, 512), (1, 20, 3, 512, 512)
        # print(f"Batch {i+1} sound_losses: ", sound_losses)  # Batch 1 sound losses: [0.327, 0.211, 0.345, 0.289, 0.324, 0.212, 0.298, 0.364, 0.432, 0.239, 0.455, 0.313, 0.276, 0.198, 0.279, 0.315, 0.323, 0.292, 0.311, 0.388]
        mse_losses += sum(sound_losses)  # mse_losses는 에폭 단위
        
    #### wandb 저장용 ####
    avg_loss = mse_losses / (len(train_loader) * num_ss)
    wandb.log({
        "train_loss(MSE)": avg_loss,
        "epoch": epoch+1
    })  
    
    ##### 모델, Json 저장 ####
    if (epoch + 1) % 5 == 0:      
        model_save_path = f"/home/mihyun/server_data/model/model2/cross_attention_output/train_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_save_path)

        print(f"Model saved at epoch {epoch+1} to {model_save_path}")

####################### validation #######################
    model.eval()
    val_mse_losses = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue 
            rgb, depth, sound_tensor = batch # (B, 3, H, W), (B, 1, H, W), (B, 20, 3, H, W)
            val_sound_losses = []

            for sound_idx in range(sound_tensor.size(1)): 
                one_sound = sound_tensor[:, sound_idx]  # (B, 3, H, W)
                
                # inputs = torch.cat([rgb, one_sound], dim=1)   # (B, 6, H, W)
                inputs, one_sound, targets = rgb.to(device), one_sound.to(device), depth.to(device)
                outputs = model(inputs, one_sound)

                mse_loss = criterion_mse(outputs, targets) 
                val_sound_losses.append(mse_loss.item())       
            
            # print(f"Batch(val) {i+1} sound_losses: ", val_sound_losses)  # Batch 1 sound losses: [0.327, 0.211, 0.345, 0.289, 0.324, 0.212, 0.298, 0.364, 0.432, 0.239, 0.455, 0.313, 0.276, 0.198, 0.279, 0.315, 0.323, 0.292, 0.311, 0.388]
            val_mse_losses += sum(val_sound_losses)  # mse_losses는 에폭 단위


    #### wandb 저장용 ####
    val_avg_loss = val_mse_losses / (len(val_loader) * num_ss)
    wandb.log({
        "val_loss(MSE)": val_avg_loss,
        "epoch": epoch+1
    })  
        
    if val_avg_loss < best_mse_loss:
        best_mse_loss = val_avg_loss
        count = 0
        model_path = f"/home/mihyun/server_data/model/model2/cross_attention_output/val_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, model_path)
    else:
        count += 1
        if count >= early_stop_num:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
