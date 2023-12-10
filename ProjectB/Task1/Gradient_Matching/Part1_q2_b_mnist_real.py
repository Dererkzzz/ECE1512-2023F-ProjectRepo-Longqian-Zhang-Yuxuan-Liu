import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug



import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler
import argparse

import time
import copy
from torchvision.utils import save_image

from scipy.io import savemat
from scipy.io import loadmat
import seaborn as sns


def main():

    class Args:
    	method = 'DC'
    	dataset = 'MNIST'
    	model = 'ConvNet'
    	ipc = 10
    	eval_mode = 'S'
    	num_exp = 1
    	num_eval = 100
    	epoch_eval_train = 50
    	Iteration = 100
    	lr_img = 0.1
    	lr_net = 0.01
    	batch_real = 256
    	batch_train = 256
    	init = 'real'
    	dsa_strategy = 'None'
    	data_path = 'data'
    	save_path = 'result'
    	dis_metric = 'ours'

    args = Args()

    
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []



    for exp in range(args.num_exp):
        

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):
            
            if it in eval_it_pool:
            	for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                    args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 50
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval,
                                                        args.ipc)  # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 50  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 50

                    accs = []
                    # args.num_eval
                    for it_eval in range(100):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(
                            args.device)  # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                            label_syn.detach())  # avoid any unaware modification
                        net_eval, acc_train, acc_test, precision, recall, f1 = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval,
                                                                 testloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                    len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration:  # record the final results
                        accs_all_exps[model_eval] += accs
                    
        
        
                
            
                
            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


            for ol in range(args.outer_loop):
                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c


                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break


                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)

            loss_avg /= (num_classes*args.outer_loop)

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))
                ''' Test '''
                accs = []
                image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                    label_syn.detach())  # avoid any unaware modification
                net_eval, acc_train, acc_test, precision, recall, f1 = evaluate_synset(1, net, image_syn_eval, label_syn_eval, testloader, args)
                accs.append(acc_test)
    
    
    result_path = os.path.join(args.save_path, 'MNIST_result')
    

    os.makedirs(result_path, exist_ok=True)
    for i in range(10):  
        os.makedirs(os.path.join(result_path, str(i)), exist_ok=True)
    mean = [0.1307]
    std = [0.3081]
    
    # save image
    for i in range(len(image_syn)):
        img = image_syn[i].cpu().detach()

        
        img = img.mul(std[0]).add(mean[0])
        
        
        
        img.clamp_(0, 1)
    
        
        img = (img * 255).to(torch.uint8)  
        
        img = transforms.functional.to_pil_image(img)
        label = int(label_syn[i].item())

        
        save_path = os.path.join(result_path, str(label), f'image_{i}.png')
        img.save(save_path)
    
    # images, labels = next(iter(data_loader_train))
    dataset_syn = torchvision.utils.make_grid(image_syn, nrow=10).cpu()
    dataset_syn = dataset_syn.numpy().transpose(1,2,0)
    dataset_syn = dataset_syn*std+mean
    print([int(label_syn[i]) for i in range(100)])

    plt.figure(dpi=300,figsize=(2,2))
    plt.axis('off')
    plt.imshow(dataset_syn)
    
    plt.savefig('MNIST_syn.jpg',dpi=300)
    


if __name__ == '__main__':
    main()


