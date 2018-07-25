import os, glob
import pandas as pd 
from tqdm import tqdm

namelist = []
idlist = []
lfwlist = []
ytflist = []

path = '/home/super/datasets/MsCeleb-modify/Top1M_MidList.Name.tsv'
lfw = '/home/super/datasets/VGGFace2/lfw.txt'
ytf = '/home/super/datasets/VGGFace2/youtube.txt'
msceleb_path = '/home/super/datasets/MsCeleb-modify/ms-celeb-orig-align/'
lfwoverlapdir = '/home/super/datasets/MsCeleb-modify/lfwoverlap-msceleb-orig/'

with open(lfw, 'r') as lfwf:
	lfw_data = lfwf.readlines()
for i in lfw_data:
	i = i.strip('\n')
	i = ' '.join(i.split('_'))
	lfwlist.append(i)

with open(ytf, 'r') as ytff:
	ytf_data = ytff.readlines()
for i in ytf_data:
	i = i.strip('\n')
	i = ' '.join(i.split('_'))
	ytflist.append(i)

with open(path, 'r') as f:
	data = f.readlines()
for entry in data:
	id_, name = entry.split('\t')
	if '@en' in name:
		name = name.split('"')[1]
		namelist.append(name)
		idlist.append(id_)

print(len(idlist))
print(len(namelist))

for i in tqdm(range(len(namelist))):
	for j in ytflist:
		if namelist[i] == j:
			os.system("mv /home/super/datasets/MsCeleb-modify/ms-celeb-orig-align/{} /home/super/datasets/MsCeleb-modify/lfwoverlap-msceleb-orig/{}".format(idlist[i],idlist[i]))
			print(namelist[i], idlist[i], os.path.isdir(msceleb_path+idlist[i]))
