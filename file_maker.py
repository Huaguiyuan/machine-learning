import numpy as np

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
#print(x,y,z)
#kbt = []
k = []
t = []
param = []
#delta = []
#nt = []
#U = []
#V = []
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
#m_x = np.vstack(x)
#m_y = np.vstack(y)
#m_z = np.vstack(z)

#print(m_z,m_y,m_x)