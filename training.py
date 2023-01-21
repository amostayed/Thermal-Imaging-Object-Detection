# TODO: tensorboard logging

'''load packages'''
from __future__ import print_function
from __future__ import division
#
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import math
#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
#
import argparse
#
from yaml import load, Loader

'''load classes and functions'''
from src.networks.yolo import Yolo
from src.datasets.dataset import ThermalDataset
from src.training.yoloLoss import yoloLoss 
#
from src.datasets.utils import image_weights_from_label_file
#
from src.postprocessing.predictions import Prediction
from src.postprocessing.evaluation import *


''' device assignment '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# utility functions
def evaluate(configs, model, prediction, loader):

	model.eval()

	labels = []
	sample_metrics = []

	for batch_i, (imgs, gt_boxes, _, _, _) in enumerate(loader):
		gt_boxes = gt_boxes.to(device)
		imgs = imgs.to(device)

        # extract labels
		labels += gt_boxes[:, -1].tolist()

		with torch.no_grad():
			pred = model(imgs)

        # filter out the raw outputs via NMS
		outputs = prediction(pred)
		sample_metrics += get_batch_statistics(outputs, gt_boxes, iou_threshold = configs['iou_threshold'])

    # concat sample statistics
	true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
	_, _, AP, _, _ = ap_per_class(true_positives, pred_scores, pred_labels, labels)

	return AP.mean()

def device_assignment(tensor_list, device):
	if isinstance(tensor_list, list):
		tensor_list = [tensor.to(device) for tensor in tensor_list]
	else:
		tensor_list = tensor_list.to(device)
	return tensor_list


def print_lr(optimizer):
	for param_group in optimizer.param_groups:
		print("Current learning rate is: {}".format(param_group['lr']))

########################
def parse_args():
    parser = argparse.ArgumentParser()
    # the config file should have all the info
    parser.add_argument('--config-path',  type = str,   default = './configurations/config.yml', help = 'configuration yaml file path')

    return parser.parse_args()


def validate(configs, model, loader, prediction):

	return evaluate(configs, model, prediction, loader)
	

def train_one_epoch(configs, model, optimizer, loader, criterion, epoch):
	print(f"Begin Epoch {epoch}")
	print_lr(optimizer)

	model.train()

	total_loss = 0.0
	'''strat training'''
	start = time.time()
	for batch_idx, (inputs, _, targets, obj_mask, no_obj_mask) in enumerate(loader):
		inputs, targets = device_assignment(inputs, device), device_assignment(targets, device)
		obj_mask, no_obj_mask = device_assignment(obj_mask, device), device_assignment(no_obj_mask, device)

		preds = model(inputs)
		optimizer.zero_grad()
		loss, _ = criterion(epoch, preds, targets, obj_mask, no_obj_mask)
		total_loss += loss.item()

		loss.backward()
		optimizer.step()
		end = time.time()

		'''print out some stats at regular interval '''
		if (batch_idx + 1) % (len(loader) // configs['print frequency']) == 0:
			print ('Epoch [%d/%d], Iter [%d/%d], Current Loss: %.4f, average_loss: %.4f, Elapsed time: %.3f seconds' 
				%(epoch + 1, configs['epochs'], batch_idx + 1, len(loader), loss.item(), total_loss / (batch_idx + 1), end - start))


def run_training():
	# parse arguments
    args = parse_args()
    with open(args.config_path, 'r') as f:
    	configs = load(f, Loader=Loader)

    #print(configs)

    # datasets and loaders
    print('Generating training and validation sets ... ')
    Transform = transforms.Compose([transforms.ToTensor()])    

    train_set = ThermalDataset(root = os.path.join(configs['train path'], 'data'), 
    						   list_file = os.path.join(configs['annot path'], configs['train fname']), 
                               image_size = configs['image size'], 
                               anchors = configs['anchors'], 
                               num_classes = configs['num class'],
                               train = True, 
                               transform = Transform)

    class_weights = np.array(configs['cls weights'])
    weights = image_weights_from_label_file(os.path.join(configs['annot path'], configs['train fname']), 
    	                                    classes_weights = class_weights, num_classes = configs['num class'])

    train_sampler = WeightedRandomSampler(weights = weights, num_samples = len(train_set), replacement = False)
    #
    train_loader = DataLoader(train_set, 
	                          batch_size = configs['batch size'], 
	                          drop_last = True,
	                          #shuffle = True,
	                          sampler = train_sampler,
	                          collate_fn = train_set.collate_fn,
	                          num_workers = configs['num workers'])

    val_set = ThermalDataset(root = os.path.join(configs['val path'], 'data'), 
                              list_file = os.path.join(configs['annot path'], configs['val fname']), 
                              image_size = configs['image size'], 
                              anchors = configs['anchors'],
                              num_classes = configs['num class'],
                              train = False, 
                              transform = Transform)
    
    val_loader = DataLoader(val_set, 
	                        batch_size = configs['batch size'],
	                        drop_last = False,
	                        collate_fn = val_set.collate_fn,
	                        shuffle = False, 
	                        num_workers = configs['num workers'])

    print('the training dataset has %d images' % (len(train_set)))
    print('the validation dataset has %d images' % (len(val_set)))


    
    # network
    print('Loading model ...')
    net = Yolo(anchors = configs['anchors'], 
    	       num_classes = configs['num class'],
               cls_probs = torch.Tensor(configs['category frequency']), 
               inp_dim = configs['image size'])

    if configs['load path']:
    	net.load_pretrained_weight(os.path.join(configs['load path'], configs['load from'] + '.pth'))
    net = net.to(device = device, dtype = torch.float32)     


    # loss & optimizer
    criterion = yoloLoss(total_epochs = configs['epochs'],
    	                 num_classes = configs['num class'], 
                         image_size = configs['image size'], 
                         anchors = configs['anchors'], 
                         l_coord = configs['l_coord'], 
                         l_obj = configs['l_obj'], 
                         l_noobj = configs['l_noobj'],
                         l_cls = configs['l_cls'],
                         l_loc = configs['l_loc'])

    optimizer = optim.SGD(net.parameters(), lr = configs['LR'], momentum = configs['momentum'], weight_decay = configs['weight decay'])


    # LR scheduling
    def schedule_LR(epoch):
    	if epoch < 50: return 1.0
    	if epoch >= 50 and epoch < 90: return 0.5
    	if epoch >= 90: return 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = schedule_LR)

    # evaluation
    prediction = Prediction(anchors = configs['anchors'],
                            beta = configs['beta'], 
                            inp_dim = configs['image size'], 
                            num_classes = configs['num class'],  
                            obj_thres = configs['obj_thres'],
                            conf_thres = configs['conf_thres'],
                            loc_thres = configs['loc_thres'],
                            sigma = configs['sigma'],
                            nms_thres = configs['nms_thres'], 
                            top_k = configs['top_k'], 
                            CUDA = torch.cuda.is_available())


    # start training loop
    print('Start training ...')

    best_map = 0.
    for epoch in range(configs['epochs']):
    	# runs the training for one epoch
    	# and prints out the loss value
    	train_one_epoch(configs, net, optimizer, train_loader, criterion, epoch)
    	# evaluate the validation set 
    	# and save the model with best mAP
    	mAP = validate(configs, net, val_loader, prediction)
    	print('Current mAP: %.2f' %(mAP * 100))

    	# save best model
    	if best_map < mAP:
    		best_map = mAP
    		print('Best avg. precision %.5f' % best_map)
    		torch.save({'model_weights': net.state_dict(), 'epoch': epoch, 'mAP': mAP}, 
    			       os.path.join(configs['save path'], configs['save name'] + f"_epoch_{epoch}.pth"))

    	# finally update the LR scheduler
    	scheduler.step()


if __name__ == "__main__":                    
    run_training()