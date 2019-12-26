import numpy as np
import cv2
import os
import torch
import glob
import torchvision.transforms as tr
from models.networks import get_nets
from albumentations import CenterCrop, PadIfNeeded, Compose
from torch.autograd import Variable

def predict_one(inp_path, outp_path, model):
	img_transforms = tr.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
	pad_transform = Compose([PadIfNeeded(736, 1280)])
	crop = CenterCrop(720, 1280)
 
	img = cv2.imread(inp_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = pad_transform(image=img)['image']
	img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32'))
	img_tensor = img_transforms(img_tensor)
 
	with torch.no_grad():
		img_tensor = Variable(img_tensor.unsqueeze(0).cuda())
		res = model(img_tensor)
	res = res[0].cpu().float().numpy()
	res = (np.transpose(res, (1, 2, 0)) + 1) / 2.0 * 255.0
	res = crop(image=res)['image']
	cv2.imwrite(outp_path, res.astype('uint8'))

def predict(inp_path, outp_path, weights_path):
	model, _ = get_nets(config['model'])
	model.load_state_dict(torch.load(weights_path)['model'])
 
	if os.path.isfile(inp_path):
		predict_one(model, inp_path, outp_path)
	else:
		if not os.path.exists(outp_path):
			os.makedirs(outp_path)
		for img in glob.glob(inp_path + '**/*', recursive=True):
			predict_one(model, img, outp_path + os.path.basename(img))