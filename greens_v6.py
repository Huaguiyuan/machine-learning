import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import os
from notify_run import Notify
#from funcao import G11 




class XorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,50)
        self.fc2 = nn.Linear(50,80)
        self.fc3 = nn.Linear(80,50)
        self.fc4 = nn.Linear(50,40)
        self.fc5 = nn.Linear(40,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = self.fc4(x)
        #x=self.fc2(x)
        x=self.fc5(x)
        return x

def load_checkpoint(model, optimizer, filename="checkpoint.pt"):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch#, losslogger



m = XorNet()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(m.parameters(), lr=1e-5)
f = open("gap_ML90.dat","w+")
training_epochs = 6000
minibatch_size = 64
notify = Notify()
#notify.register()

#
#pairs = [np.asarray([x1,x2]), [y]] 
# input-output pairs
# pairs = [(np.asarray([4.0,0.85]), [0.072]),
#           (np.asarray([6.0,0.75]), [0.1]),
#           (np.asarray([8.0,0.65]), [0.15]),
#           (np.asarray([10.0,0.55]), [0.312]),
#           (np.asarray([3.0,0.8]), [0.068])]
f0 = open("orderparn85U8t1tl0.4v0q.dat", "r")
f1 = open("orderparn85U8t1tl0.4v01q.dat", "r")
f2 = open("dorderparn85U8t1tl0.4v02q.dat", "r")
f3 = open("dorderparn85U8t1tl0.4v03q.dat", "r")
f4 = open("dorderparn85U8t1tl0.4v04q.dat", "r")
x0,y0 = np.loadtxt(f0,delimiter=' ',unpack=True)
x1,y1 = np.loadtxt(f1,delimiter=' ',unpack=True)
x2,y2 = np.loadtxt(f2,delimiter=' ',unpack=True)
x3,y3 = np.loadtxt(f3,delimiter=' ',unpack=True)
x4,y4 = np.loadtxt(f4,delimiter=' ',unpack=True)
f0.close()
f1.close()
f2.close()
f3.close()
f4.close()

#print(x,y,z)
#kbt = []
k = []
t = []
param = []
#delta = []
#nt = []
#U = []
#V = []
#print*,
conta = 0
while(conta<=499): 
    k.append([x0[conta],0.85,8.0,0.0])
    t.append([y0[conta]])
    conta+=1
conta = 0
while(conta<=499): 
    k.append([x1[conta],0.85,8.0,0.1])
    t.append([y1[conta]])
    conta+=1
conta = 0
while(conta<=499): 
    k.append([x2[conta],0.85,8.0,0.2])
    t.append([x2[conta]])
    conta+=1
conta = 0
while(conta<=499): 
    k.append([x3[conta],0.85,8.0,0.3])
    t.append([y3[conta]])
    conta+=1
conta = 0
while(conta<=499): 
    k.append([x4[conta],0.85,8.0,0.4])
    t.append([y4[conta]])
    conta+=1
#k.append(y)
m_k = np.vstack(k)
m_t = np.vstack(t)
#x.append(y)
print(m_k)
print(m_t)

f0.close()
f1.close()
f2.close()
f3.close()
f4.close()
#x_input = torch.Tensor(5)
#y_input = torch.Tensor(5)
#x1_input = [2,2]
#x2_input = [2]
# x_input = []
# y_input = []
# ur = 3.0
# nt = 0.3
# conta = 0
# indice = 0
# while(conta<=5):
#     nt = 0.3
#     #nt = nt+0.1
#     #y_input.append(G11(nt,ur,12.0))
#     while(indice<=5):
#         x_input.append([ur,nt]) 
#         #ur = ur+1.0
#         y_input.append(G11(nt,ur,12.0))
#         #print(ur,nt,G11(nt,ur,12.0))
#         nt = nt+0.1
#         indice+=1
#     indice=0
#     ur = ur+1.0
#     conta+=1
    
#x_input.append(x1_input)
#x_input.append(x2_input)
#print(x_input,y_input) 
#read()
#print(y_input)
#x_tensor = torch.Tensor(x1_input)
#y_tensor = torch.Tensor(y_input)

#print(x_tensor)
#print(y_tensor)

#print(x1_input)
#state_matrix = np.vstack([z[0] for z in pairs])
#label_matrix = np.vstack([z[1] for z in pairs])
#state_matrix = np.vstack(x_input)
#label_matrix = np.vstack(y_input)
state_matrix = np.vstack(m_k)
label_matrix = np.vstack(m_t)
#print(state_matrix,label_matrix)
#print(label_matrix)
i = 0
m, optimizer, i = load_checkpoint(m, optimizer, "checkpoint.pt")
#i = start_epoch
print(i)
#stop
i = training_epochs + 50
while i < training_epochs:
    print(i)   
    for batch_ind in range(500*5):
        #print(batch_ind)
         # wrap the data in variables
        minibatch_state_var = Variable(torch.Tensor(state_matrix))
        minibatch_label_var = Variable(torch.Tensor(label_matrix))
        
        # forward pass
        y_pred = m(minibatch_state_var)
        
        # compute and print loss
        loss = loss_fn(y_pred, minibatch_label_var)
       # print(i, batch_ind, loss.data[0])

        # reset gradients
        optimizer.zero_grad()
        
        # backwards pass
        loss.backward()
        
        # step the optimizer - update the weights
        optimizer.step()
    if i % 25 == 0 and i > 1:
        state = {'epoch': i + 1, 'state_dict': m.state_dict(),'optimizer': optimizer.state_dict()}
        torch.save(state,"checkpoint.pt")
        notify.send("salvo",i)
        print("salvo",i)
    i+=1

#print("Function after training:")
#print("f(4.5,0.55) = {}".format(m(Variable(torch.Tensor([4.5,0.55]).unsqueeze(0)))))
#print("f(4.0,0.5) = {}".format(m(Variable(torch.Tensor([4.0,0.5]).unsqueeze(0)))))

contador = 0
#ur = 3.0
#nt = 0.3
temp = 0.000001
while(contador<=1000):
    y = m(Variable(torch.Tensor([temp,0.85,8.0,0.2]).unsqueeze(0)))
    z = y.item()
    formato = "{} {}\n"
    #print(ur,nt,z,file=green.dat)
    #f.write("{}, {}, {}".format(m(Variable(torch.Tensor([ur,nt]).unsqueeze(0)))))
    f.write(formato.format(temp,z))
    #ur = ur+0.05
    #nt = nt+0.005
    temp +=0.005
    contador+=1
f.close()
print(":)")
notify.send("acabou de rodar")
#print("f(1,0) = {}".format(m(Variable(torch.Tensor([6.0,0.7]).unsqueeze(0)))))



