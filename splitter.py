import numpy as np
import glob
import os
import shutil

flist = glob.glob('labelled_plumes/*/*/*npz')

for fname in flist:
	mask = np.load(fname)['mask']
	cloudness = np.load(fname)['channels'][:, :, -1]
	cloudness = np.sum(cloudness >= 60)/(cloudness.shape[0]*cloudness.shape[1])*100
	folder = '/'.join(fname.split('/')[:-1])
	parent = folder.split('/')[-2]
	imgfolder = folder.split('/')[-1]
	print('Percent of cloud score > 60: ', cloudness)

	if np.any(mask != 0):
		if not os.path.exists('classified/positive/'+parent):
			os.makedirs('classified/positive/'+parent)
		if not os.path.exists('classified/positive/'+parent+'/'+imgfolder):
			shutil.copytree(folder, 'classified/positive/'+parent+'/'+imgfolder)
	else:
		if cloudness >= 50:
			if not os.path.exists('classified/cloudy/'+parent):
				os.makedirs('classified/cloudy/'+parent)
			if not os.path.exists('classified/cloudy/'+parent+'/'+imgfolder):
				shutil.copytree(folder, 'classified/cloudy/'+parent+'/'+imgfolder)
		else:
			if not os.path.exists('classified/negative/'+parent):
				os.makedirs('classified/negative/'+parent)
			if not os.path.exists('classified/negative/'+parent+'/'+imgfolder):
				shutil.copytree(folder, 'classified/negative/'+parent+'/'+imgfolder)