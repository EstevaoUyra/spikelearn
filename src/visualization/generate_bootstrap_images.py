import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def permute(matrix, line):
    matrix[line,:] = np.random.permutation(matrix[line,:])

def matsave_img(matrix,index):
    plt.figure(figsize=(5,10))
    sns.heatmap(matrix,cmap='plasma',annot=True,cbar=False,vmin=1,vmax=10)
    plt.axis('off');
    plt.text(-.8,10,'Trial',rotation=90, fontsize=20)
    plt.text(3.3,20.8,'Time bin',fontsize=20)
    plt.savefig('/home/tevo/Documents/UFABC/SingleUnit Spike Learning/reports/Presentation_241117/bootstrap/shuffle_bins%d.png'%(index+1),transparent=True)

plt.imshow(np.ones((1,10)).cumsum(axis=1),cmap='plasma')
plt.axis('off')
plt.savefig('/home/tevo/Documents/UFABC/SingleUnit Spike Learning/reports/Presentation_241117/bootstrap/onetrial.png',transparent=True,dpi=100)

plt.figure(figsize=(10,1))
sns.heatmap(np.ones((1,10)).cumsum(axis=1),cmap='plasma',annot=True,cbar=False,vmin=1,vmax=10)
plt.axis('off')
plt.savefig('/home/tevo/Documents/UFABC/SingleUnit Spike Learning/reports/Presentation_241117/bootstrap/onetrialLabel.png',transparent=True,dpi=200)

plt.figure(figsize=(5,1))
sns.heatmap(np.ones((1,10)).cumsum(axis=1)[:,2:-3],cmap='plasma',annot=True,cbar=False,vmin=1,vmax=10)
plt.axis('off')
plt.savefig('/home/tevo/Documents/UFABC/SingleUnit Spike Learning/reports/Presentation_241117/bootstrap/onetrialcut.png',transparent=True,dpi=100)

time = np.ones((15,10)).cumsum(axis=1)[:,2:-3]
print(time)
matsave_img(time,-1)

for i in range(15):
    permute(time,i)
    matsave_img(time,i)
