import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
import pandas as pd
from termcolor import colored
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_curve,auc,roc_auc_score
import os
# import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score, matthews_corrcoef
import math

parser = argparse.ArgumentParser(description='embeddings_for_RBP_prediction')
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--model_dir', default='Model/', help='model directory')
parser.add_argument('--rep_dir', help='represention file directory')
parser.add_argument('--pro_label_dir', help='pro_label file directory')
parser.add_argument('--load_model_dir', default=None,help='trained model file directory')
parser.add_argument('--big_or_small_model',type=int,default=0, help='choose between big and small model,0 means big')
parser.add_argument('--learning_rate',type=float,default=0.0001, help='learning rate')
parser.add_argument('--batch_size',type=int,default=1024)
args = parser.parse_args()

rep_all_pd=pd.read_csv(args.rep_dir)
pro=pd.read_csv(args.pro_label_dir)
label=torch.tensor(pro['label'].values)
head,tail=os.path.split(args.pro_label_dir)
trP=tail.split('trP')[1].split('_')[0]
trN=tail.split('trN')[1].split('_')[0]
vaP=tail.split('VaP')[1].split('_')[0]
vaN=tail.split('VaN')[1].split('_')[0]
teP=tail.split('TeP')[1].split('_')[0]
teN=tail.split('TeN')[1].split('_')[0]
data=torch.tensor(rep_all_pd.values)
print(trP,trN,vaP,vaN,teP,teN)
# print(data.shape,label.shape)
print(label.shape,data.shape)
train_data,train_label=data[:int(trP)+int(trN)].double(),label[:int(trP)+int(trN)]
test_data,test_label=data[int(trP)+int(trN):-int(teP)-int(teN)].double(),label[int(trP)+int(trN):-int(teP)-int(teN)]
# LOSS_WEIGHT_POSITIVE = math.sqrt((int(trP)+int(trN)) / (2.0 * int(trP)) )
# LOSS_WEIGHT_NEGATIVE = math.sqrt((int(trP)+int(trN)) / (2.0 * int(trN)) )
LOSS_WEIGHT_POSITIVE = (int(trP)+int(trN)) / (2.0 * int(trP)) 
LOSS_WEIGHT_NEGATIVE = (int(trP)+int(trN)) / (2.0 * int(trN)) 
# https://towardsdatascience.com/deep-learning-with-weighted-cross-entropy-loss-on-imbalanced-tabular-data-using-fastai-fe1c009e184c
soft_max=nn.Softmax(dim=1)
# class_weights=torch.FloatTensor([w_0, w_1]).cuda()
weig=torch.FloatTensor([LOSS_WEIGHT_NEGATIVE,LOSS_WEIGHT_POSITIVE]).double().cuda()
# train_data,train_label=genData("./train_peptide.csv",260)
# test_data,test_label=genData("./test_peptide.csv",260)

train_dataset = Data.TensorDataset(train_data, train_label)
test_dataset = Data.TensorDataset(test_data, test_label)
batch_size=args.batch_size
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
Emb_dim=data.shape[1]
if not os.path.exists(args.model_dir):
   os.mkdir(args.model_dir)
head1,tail1=os.path.split(args.pro_label_dir)
if args.load_model_dir ==None:
   logits_output=os.path.join(args.model_dir,tail1.split('_')[0]+'_'+args.rep_dir.split('/')[-2]  \
      +str(args.big_or_small_model)+ '_logits.csv')
   model_loc=os.path.join(args.model_dir,tail1.split('_')[0]+'_'+args.rep_dir.split('/')[-2]   \
      +str(args.big_or_small_model)+ '.pl')
else:
    logits_output=os.path.join(args.model_dir,'fine_tune'+tail1.split('_')[0]+'_'+args.rep_dir.split('/')[-2]  \
          +str(args.big_or_small_model)+ '_logits.csv')
    model_loc=os.path.join(args.model_dir,'fine_tune'+tail1.split('_')[0]+'_'+args.rep_dir.split('/')[-2]   \
          +str(args.big_or_small_model)+ '.pl')
class newModel1(nn.Module):
    def __init__(self, vocab_size=26):
        super().__init__()
        self.hidden_dim = 256
        self.batch_size = 256
        self.emb_dim = Emb_dim
        
        # self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # self.gmlp_t=gMLP(num_tokens = 1000,dim = 32, depth = 2,  seq_len = 40, act = nn.Tanh())
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=6, 
                               bidirectional=True, dropout=0.05)
        
        
        self.block1=nn.Sequential(nn.Linear(3584,1024),
                                            nn.BatchNorm1d(1024),
                                            nn.LeakyReLU(),
                                            nn.Linear(1024,512),
                                            nn.BatchNorm1d(512),
                                            nn.LeakyReLU(),
                                            nn.Linear(512,256),
                                 )

        self.block2=nn.Sequential(
                                               nn.BatchNorm1d(256),
                                               nn.LeakyReLU(),
                                               nn.Linear(256,128),
                                               nn.BatchNorm1d(128),
                                               nn.LeakyReLU(),
                                               nn.Linear(128,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,2)
                                            )
        
    def forward(self, x):
        # x=self.embedding(x)
        # output=self.transformer_encoder(x).permute(1, 0, 2)
        # output=self.gmlp_t(x).permute(1, 0, 2)
        x=x.view(1,x.shape[0],x.shape[1])
        # output=self.gmlp_t(x).permute(1, 0, 2)
        # print(output.shape)
        output,hn=self.gru(x)
        output=output.permute(1,0,2)
        hn=hn.permute(1,0,2)
        output=output.reshape(output.shape[0],-1)
        hn=hn.reshape(output.shape[0],-1)
        output=torch.cat([output,hn],1)
        # print('output.shape',output.shape)
        output=self.block1(output)
        return self.block2(output)
class newModel2(nn.Module):
    def __init__(self, vocab_size=26):
        super().__init__()
        self.hidden_dim = 48
        self.batch_size = 256
        self.emb_dim = Emb_dim
        
        # self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # self.gmlp_t=gMLP(num_tokens = 1000,dim = 32, depth = 2,  seq_len = 40, act = nn.Tanh())
        # self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=4, 
        #                        bidirectional=True, dropout=0.2)
        self.c1_1 = nn.Conv1d(32, 256, 1)
        self.c1_2 = nn.Conv1d(32, 256, 3)
        self.c1_3 = nn.Conv1d(32, 256, 5)
        self.p1 = nn.MaxPool1d(3, stride=3)     
        self.c2 = nn.Conv1d(256, 128, 3)
        self.p2 = nn.MaxPool1d(3, stride=3)  
        self.c3 = nn.Conv1d(128, 128, 3)
        # self.p3 = nn.MaxPool1d(3, stride=1)       
        self.drop=nn.Dropout(p=0.01)
        self.block2=nn.Sequential(        
                                               nn.Linear(896,512),
                                               nn.BatchNorm1d(512),
                                               nn.LeakyReLU(),
                                               nn.Linear(512,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,2)
                                            )
        
    def forward(self, x):
        # x=self.embedding(x)
        # output=self.transformer_encoder(x).permute(1, 0, 2)
        # output=self.gmlp_t(x).permute(1, 0, 2)
        x=x.view(x.shape[0],32,32)
        # x=x.transpose(1,2)
        # output=self.gmlp_t(x).permute(1, 0, 2)
        # print(output.shape)
        c1_1=self.c1_1(x)
        c1_2=self.c1_2(x)
        c1_3=self.c1_3(x)
        c=torch.cat((c1_1, c1_2, c1_3), -1)
        # print(c1_1.shape,c1_2.shape,c1_3.shape,c.shape)
        p = self.p1(c)
        c=self.c2(p)
        p=self.p2(c)
        # print(p.shape)
        c=self.c3(p)
        # print(c.shape)
        # p=self.p3(c)

        # print(p.shape)
        # print('output.shape',output.shape)
        # print(c.shape)
        c=c.view(c.shape[0],-1)
        c=self.drop(c)
        # print(c.shape)
        return self.block2(c)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        euclidean_distance = F.pairwise_distance(output1, output2)
        # print(output1.shape,output2.shape,label.shape)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        
        return loss_contrastive
    
    
def collate(batch):
    seq1_ls=[]
    seq2_ls=[]
    label1_ls=[]
    label2_ls=[]
    label_ls=[]
    batch_size=len(batch)
    for i in range(int(batch_size/2)):
        seq1,label1=batch[i][0],batch[i][1]
        seq2,label2=batch[i+int(batch_size/2)][0],batch[i+int(batch_size/2)][1]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label=(label1*label2)+(1-label1)*(1-label2)
        # label=(label1^label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    seq1=torch.cat(seq1_ls).to(device)
    seq2=torch.cat(seq2_ls).to(device)
    label=torch.cat(label_ls).to(device)
    label1=torch.cat(label1_ls).to(device)
    label2=torch.cat(label2_ls).to(device)
    return seq1,seq2,label,label1,label2
    
train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                                  shuffle=True,collate_fn=collate)

  
device = torch.device("cuda",0)


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        x,y=x.to(device),y.to(device)
        outputs=net(x)
        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def to_log(log):
    with open("./modelLog.log","a+") as f:
        f.write(log+'\n')


def main():
  if args.big_or_small_model ==0:
     net=newModel1().double().to(device)
  else:
     net=newModel2().double().to(device)
      # state_dict=torch.load('/content/Model/pretrain.pl')
      # net.load_state_dict(state_dict['model'])
  if args.load_model_dir != None:
      state_dict=torch.load(args.load_model_dir)
      net.load_state_dict(state_dict['model'])
  # lr = 0.0001
  optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate,weight_decay=5e-4)
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5, factor=0.75,verbose=True)
# https://discuss.pytorch.org/t/reducelronplateau-not-doing-anything/24575/10
  # criterion = ContrastiveLoss()
  # criterion_model = nn.CrossEntropyLoss(reduction='sum')
  

  criterion_model = nn.CrossEntropyLoss(weight=weig,reduction='mean')
  best_bacc=0
  best_aupr=0
  EPOCH=args.epoch
  CUDA_LAUNCH_BLOCKING=1
  for epoch in range(EPOCH):
      loss_ls=[]

      t0=time.time()
      net.train()
      # for seq1,seq2,label,label1,label2 in train_iter_cont:
      for seq,label in train_iter:
              # print(seq1.shape,seq2.shape,label.shape,label1.shape,label2.shape)
              seq,label=seq.to(device),label.to(device)
              output=net(seq)
              loss=criterion_model(output,label)
  #             print(loss)
              optimizer.zero_grad() 
              loss.backward()
              optimizer.step()
              loss_ls.append(loss.item())
      lr_scheduler.step(loss)

      if epoch %100 ==0:
            torch.save({
                      'model_state_dict': net.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()
                     },  os.path.join('/content/Model', 'ckpt_{}.pl'.format(epoch)))
      net.eval() 
      with torch.no_grad(): 
          train_acc=evaluate_accuracy(train_iter,net)
          # test_acc=evaluate_accuracy(test_iter,net)
          test_data_gpu=test_data.to(device)
          test_logits=net(test_data_gpu)
          outcome=np.argmax(test_logits.detach().cpu(), axis=1)
          test_bacc=balanced_accuracy_score(test_label, outcome)
          precision, recall, thresholds = precision_recall_curve(test_label, soft_max(test_logits.cpu())[:,1])
          test_aupr = auc(recall, precision)
      results=f"epoch: {epoch+1}, loss: {np.mean(loss_ls):.5f}\n"
      # results=f"epoch: {epoch+1}\n"
      results+=f'\ttrain_acc: {train_acc:.4f}, test_aupr: {colored(test_aupr,"red")},test_bacc: {colored(test_bacc,"red")}, time: {time.time()-t0:.2f}'
      print(results)
      to_log(results)
      if test_aupr>best_aupr:
          best_aupr=test_aupr
          torch.save({"best_aupr":best_aupr,"model":net.state_dict(),'args':args},model_loc)
          print(f"best_aupr: {best_aupr}")
  state_dict=torch.load(args.load_model_dir)

# state_dict=torch.load('/content/Model/pretrain.pl')
  net.load_state_dict(state_dict['model'])
  pro=pd.read_csv(args.pro_label_dir)
  label=torch.tensor(pro['label'].values)
  # final_test_data,final_test_label=data[9655+1068:].double(),label[9655+1068:]
  # train_data,train_label=data[:6011].double(),label[:6011]
  final_test_data,final_test_label=data[-int(teP)-int(teN):].double(),label[-int(teP)-int(teN):]
  final_test_data=final_test_data.to(device)
  net.eval() 
  with torch.no_grad(): 
      logits=net(final_test_data)
  # logits_output=os.path.split(rep_file)[1].replace('.csv','_logtis.csv')
  logits_cpu=logits.cpu().detach().numpy()
  logits_cpu_pd=pd.DataFrame(logits_cpu)
  logits_cpu_pd.to_csv(logits_output,index=False)
  outcome=np.argmax(logits.cpu().detach().numpy(), axis=1)

  MCC= matthews_corrcoef(final_test_label, outcome)
  acc = accuracy_score(final_test_label, outcome)
  bacc=balanced_accuracy_score(final_test_label, outcome)
  precision1, recall1, thresholds1 = precision_recall_curve(final_test_label, soft_max(torch.tensor(logits_cpu))[:,1])
  final_test_aupr = auc(recall1, precision1)
  final_auc_roc=roc_auc_score(final_test_label, soft_max(torch.tensor(logits_cpu))[:,1])
  # final_test_aupr=0
  print('bacc,MCC,final_test_aupr,final_auc_roc')
  print(bacc,MCC,final_test_aupr,final_auc_roc)
if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING=1
    main()
