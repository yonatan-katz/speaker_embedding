#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:10:04 2019

@author: yonic
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from speaker_embed import networks
from database import tedlium
from tacotron2.layers import TacotronSTFT


SFT_CONFIG={    
    "sampling_rate": 16000,
    "filter_length": 400,
    "hop_length": 160,
    "win_length": 400,
    "mel_fmin": 0.0,
    "mel_fmax": 8000.0}

MAX_WAV_VALUE = 32768.0
EMBEDDING_SIZE = 512

class EmbeddingNetClassifier(networks.EmbeddingNet):
    def __init__(self,num_of_clusses):
        super(EmbeddingNetClassifier, self).__init__()
        self.logits = networks.EmbeddingNet()        
        self.fc = nn.Linear(EMBEDDING_SIZE,num_of_clusses)
        
    def forward(self, x): 
        x = self.logits(x)
        x = self.fc(x)
        return F.log_softmax(x)
        #return F.softmax(x)
    

def data_generator(data_base,chunk_length_in_sec,label_order_list,batch_size):    
    stft = TacotronSTFT(**SFT_CONFIG)
    keys = data_base.get_db_keys()    
    times = data_base.get_db_wv_times(keys[0])
    batch_x = []
    batch_y = []
    while True:        
            for k in keys:                
                label = np.array([label_order_list.index(k)])                 
                sampling_rate, speech = data_base.get_wav(keys[0],*times[0])
                chunks = int(len(speech)/sampling_rate/chunk_length_in_sec)
                audio_length = sampling_rate*chunk_length_in_sec
                for chunk in range(chunks):                
                    audio = speech[chunk*audio_length:(chunk+1)*audio_length]
                    audio_norm = audio / MAX_WAV_VALUE    
                    audio_norm = torch.from_numpy(audio_norm).float()
                    audio_norm = audio_norm.unsqueeze(0)
                    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
                    melspec = stft.mel_spectrogram(audio_norm)
                    mel_np = melspec.detach().numpy()
                    for i in range(mel_np.shape[1]):
                        channel_mean = np.mean(mel_np[0,i,:])  
                        mel_np[0,i,:] = mel_np[0,i,:] - channel_mean
                    
                    #normalized_mel = torch.from_numpy(mel_np)
                    batch_x.append(mel_np)
                    batch_y.append(label)
                    #yield normalized_mel.unsqueeze(1), Variable(y_tensor)
                    if len(batch_x) >= batch_size:
                        x = torch.from_numpy(np.array(batch_x))
                        y = Variable(torch.from_numpy(np.concatenate(batch_y)).long())
                        batch_x = []
                        batch_y = []
                        yield x,y
                        
                    
    

def main(): 
    torch.cuda.init()
    device = torch.cuda.current_device()    
    torch.cuda.set_device(device)
    batch_size = 64
    chunk_length_in_sec=3
    learning_rate = 1e-2
    data_base = tedlium.TedLium(mode='train')
    label_order_list = sorted(data_base.get_db_keys())
    num_of_clusses = len(set(label_order_list))
    print('Tedium DB num of classes is:',num_of_clusses)
    G_data = data_generator(data_base=data_base,
        chunk_length_in_sec=chunk_length_in_sec,
        label_order_list=label_order_list,
        batch_size=batch_size)
    
    embedding_net = EmbeddingNetClassifier(num_of_clusses=num_of_clusses).to('cuda:0')    
    optimizer = torch.optim.Adam(embedding_net.parameters(), 
        lr=learning_rate)
    criterion = nn.NLLLoss()    
    
    batch_idx = 0        
    for batch_idx in range(1000):
        data,target = next(G_data)        
        data = data.to('cuda:0')
        target = target.view(-1).to('cuda:0')
        optimizer.zero_grad()
        net_out = embedding_net(data)        
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        batch_idx += 1
        if batch_idx % 10 == 0:
            print('Loss:',loss.data)  

    
if __name__ == '__main__':
    main()

    
    