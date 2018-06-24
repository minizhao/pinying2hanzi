
# coding: utf-8

# In[7]:


import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import unidecode
import sys


# In[8]:


# Hyper Parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 10
num_samples = 1000   # number of words to be sampled
seq_length = 30
learning_rate = 0.002


# In[9]:


class data_loader(object):
    def __init__(self):
        #得到训练数据
        #unidecode会把汉语句子转换为拼音
        self.file_pingyin = unidecode.unidecode(open('sents.csv').read())
        self.file_hanzi = open('sents.csv').read()

        #先只去前面20w数据吧,
        self.input_sents=[line.strip() for line in self.file_pingyin.split('\n')][:200000]
        self.target_sents=[line.strip() for line in self.file_hanzi.split('\n')][:200000]
        #释放掉文件内存
        del self.file_pingyin
        del self.file_hanzi
        self.i=0

    def shuffle(self):
        #打乱文本数据
        shuff_idxs=np.random.permutation(len(self.target_sents))
        self.input_sents=np.array(self.input_sents)[shuff_idxs]
        self.target_sents=np.array(self.target_sents)[shuff_idxs]


    def count_word_nums(self):
        pingyin=[]
        hanzi=[]
        for line in self.input_sents:
            for w in line.split():
                if w not in pingyin:pingyin.append(w)
        for line in self.target_sents:
            for w in line.split():
                if w not in hanzi:hanzi.append(w)
        self.pingyin=pingyin
        self.hanzi=hanzi
        self.pingyin_nums=len(pingyin)
        self.hanzi_nums=len(hanzi)

    def random_sample(self):
        while True:
            rdm_inp=[self.pingyin.index(x) for x in self.input_sents[self.i].split()]
            rand_tag=[self.hanzi.index(x) for x in self.target_sents[self.i].split()]
            self.i=(self.i+1)%len(self.target_sents)
            if len(rdm_inp)>0 and len(rand_tag)>0:
                break
        rdm_inp=Variable(torch.from_numpy(np.array(rdm_inp))).cuda()
        rand_tag=Variable(torch.from_numpy(np.array(rand_tag))).cuda()
        return (rdm_inp,rand_tag)

    def longtensor_to_string(self,t):
        s = ''
        for i in range(t.size(0)):
            top_i = t.data[i]
            s += self.pingyin[top_i]+" "
        return s

    def tensor_to_string(self,t):
        s = ''
        for i in range(t.size(0)):
            ti = t[i]
            top_k = ti.data.topk(1)
            top_i = top_k[1][0]
            s += self.hanzi[top_i]+" "
        return s


# In[10]:


class RNNLM(nn.Module):
    def __init__(self, py_vocab_size,hz_vocab_size,embed_size, hidden_size, num_layers=1):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(py_vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, hz_vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        # Forward propagate RNN
        out, h = self.lstm(x, h)
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        # Decode hidden states of all time step
        out = self.linear(out)
        return out, h


# In[11]:

def save(model):
    save_filename = 'pin2han.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

# In[12]:


if __name__ == '__main__':
    print('loding data')
    data_loader=data_loader()
    data_loader.count_word_nums()
    data_loader.shuffle()
    print('start training')

    py_vcoab_size=data_loader.pingyin_nums
    hz_vcoab_size=data_loader.hanzi_nums

    rnnlm=RNNLM(py_vcoab_size,hz_vcoab_size,embed_size,hidden_size).cuda()
    #rnnlm = torch.load('pin2han.pt').cuda()
    print(rnnlm)
    data_loader.random_sample()
    optimizer = torch.optim.Adam(rnnlm.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10000000):
        states = (Variable(torch.zeros(num_layers, 1, hidden_size)).cuda(),
                  Variable(torch.zeros(num_layers, 1, hidden_size)).cuda())
        input_x,target=data_loader.random_sample()
        out,h=rnnlm(input_x.view(1,-1),states)
        rnnlm.zero_grad()
        loss=criterion(out,target.view(-1))
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print('epoch : '+str(epoch)+' loss:'+str(loss.data[0]))
            print('(input) "%s"' % data_loader.longtensor_to_string(input_x))
            print('(generated) "%s"' % data_loader.tensor_to_string(out))
            save(rnnlm)
