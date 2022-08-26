import numpy as np
import glob
import os
import shutil

flist = glob.glob('labelled_plumes/*/*/*npz')

for fname in flist:
	mask = np.load(fname)['mask']
	folder = '/'.join(fname.split('/')[:-1])
	parent = folder.split('/')[-2]
	imgfolder = folder.split('/')[-1]

	if np.any(mask != 0):
		if not os.path.exists('positive/'+parent):
			os.makedirs('positive/'+parent)
		if not os.path.exists('positive/'+parent+'/'+imgfolder):
			shutil.copytree(folder, 'positive/'+parent+'/'+imgfolder)
	else:
		if not os.path.exists('negative/'+parent):
			os.makedirs('negative/'+parent)
		if not os.path.exists('negative/'+parent+'/'+imgfolder):
			shutil.copytree(folder, 'negative/'+parent+'/'+imgfolder)