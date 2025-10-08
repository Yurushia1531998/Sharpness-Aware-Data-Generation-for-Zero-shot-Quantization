import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
import copy
from utils import ActivationHook,BackwardHook, find_parent
from SAM import SAM
from reconstruct import quantize_model, reconstruct, reconstruct_2, QuantizableLayer,WeightQuantizer,ActivationQuantizer
from reconstruct import get_all_optim, enable_training_mode
from torch.utils.data import DataLoader
from pynvml import *
from collections import OrderedDict
import time
import random
nvmlInit()
log = logging.getLogger(__name__)


class Generator(nn.Module):
    def __init__(self, image_size=224, latent_dim=256, c1=128, c2=64):
        super(Generator, self).__init__()

        self.init_size = image_size // 4
        self.linear = nn.Linear(latent_dim, c1 * self.init_size ** 2)
        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(c1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c1, c2, 3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c2, 3, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(3, affine=False)
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        out = self.conv_layers(out)
        return out

class SwingConv2d(nn.Module):
    def __init__(self, org_module, jitter_size=1):
        super(SwingConv2d, self).__init__()
        self.org_module = org_module
        self.jitter_size = jitter_size

    def forward(self, x):
        src_x = np.random.randint(self.jitter_size*2+1)
        src_y = np.random.randint(self.jitter_size*2+1)
        input_pad = F.pad(x, [self.jitter_size for i in range(4)], mode='reflect')
        input_new = input_pad[:, :, src_y:src_y+x.shape[2], src_x:src_x+x.shape[3]] 
        assert input_new.shape == x.shape, f'{input_new.shape}, {input_pad.shape}, {x.shape}'
        return self.org_module(input_new)

def l2_loss(A, B):
    return (A - B).norm()**2 / B.size(0)


def distill_data(model, batch_size, total_samples, lr_g=0.1, lr_z=0.01, iters=4000):
    """Generate synthetic dataset using distillation

    Args:
        model: model to be distilled
        batch_size: batch size at distillation
        total_samples: # of images to generate
        lr_g: lr of generator
        lr_z: lr of latent vector
        iters: # of iterations per distillation batch.

    Returns:
        Tensor: synthetic dataset (dim: total_samples x 3 x 224 x 224)
    """
    latent_dim = 256
    eps = 1e-6

    model = copy.deepcopy(model).cuda().eval()

    hooks, bn_stats = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.stride != (1, 1):
                parent = find_parent(model, name)
                setattr(parent, name.split('.')[-1], SwingConv2d(module, jitter_size=1))

        elif isinstance(module, nn.BatchNorm2d):
            hooks.append(ActivationHook(module))
            bn_stats.append((module.running_mean.detach().clone().cuda(),
                             torch.sqrt(module.running_var + eps).detach().clone().cuda()))

    dataset = []
    for i in range(total_samples // batch_size):
        log.info(f'Generate Image ({i * batch_size}/{total_samples})')
        # initialize the criterion, optimizer, and scheduler
        z = torch.randn(batch_size, latent_dim).cuda().requires_grad_()
        generator = Generator(latent_dim=latent_dim).cuda()

        opt_z = optim.Adam([z], lr=lr_g)
        scheduler_z = optim.lr_scheduler.ReduceLROnPlateau(opt_z, min_lr=1e-4, verbose=False, patience=100)
        opt_g = optim.Adam(generator.parameters(), lr=lr_z)
        scheduler_g = optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.95)

        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()

        for it in range(iters):
            model.zero_grad()
            opt_z.zero_grad()
            opt_g.zero_grad()

            x = generator(z)
            model(x)

            mean_loss, std_loss = 0, 0
            data_std, data_mean = torch.std_mean(x, [2, 3])
            mean_loss += l2_loss(input_mean, data_mean)
            std_loss += l2_loss(input_std, data_std)

            for (bn_mean, bn_std), hook in zip(bn_stats, hooks):
                bn_input = hook.inputs
                data_std, data_mean = torch.std_mean(bn_input, [0, 2, 3])
                mean_loss += l2_loss(bn_mean, data_mean)
                std_loss += l2_loss(bn_std, data_std)

            total_loss = mean_loss + std_loss
            total_loss.backward()
            opt_z.step()
            opt_g.step()
            scheduler_z.step(total_loss.item())

            if (it + 1) % 100 == 0:
                log.info(f'{it + 1}/{iters}, Loss: {total_loss:.3f}, Mean: {mean_loss:.3f}, Std: {std_loss:.3f}')
                scheduler_g.step()

        dataset.append(x.detach().clone())


    for hook in hooks:
        hook.remove()

    dataset = torch.cat(dataset)
    return dataset

def enable_update_weight(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantizableLayer):
            module.update_weight = True
def unable_update_weight(model):
    for name, module in model.named_modules():
        if isinstance(module,QuantizableLayer):
            module.update_weight = False
            if module.activation_quantizer and module.enable_act_quant:
                module.enable_act_quant=False



def test_freezed_model(net):
    for name, param in net.named_parameters():
        if (name.endswith(".weight") or name.endswith(".bias")) and param.requires_grad == True:
            print("Param ",name," is still true")
def freeze_layer_weight(net):
    for name, param in net.named_parameters():
        if name.endswith(".weight") or name.endswith(".bias"):
            param.requires_grad = False
def unfreeze_layer_weight(net):
    for name, param in net.named_parameters():
        if name.endswith(".weight") or name.endswith(".bias"):
            param.requires_grad = True
def freeze_all_weight(net):
    for param in net.parameters():

        param.requires_grad = False
def unfreeze_all_weight(net):
    for param in net.parameters():

        param.requires_grad = True


def check_gpu_memory():
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')
    print("memory allocated: ",torch.cuda.memory_allocated())
    print("max memory allocate: ", torch.cuda.max_memory_allocated())
    print(" ")
def get_reconstruction_loss(model,data,fp_predict,all_hooks,loss_type):
    output = model(data)

    loss = 0

    if "all_layer" in loss_type:

        for hook_fp, hook in zip(all_hooks[1], all_hooks[2]):
            loss += (hook_fp.outputs - hook.outputs).pow(2).sum(1).mean()

    else:
        loss = (fp_predict - output).pow(2).sum(1).mean()

    loss = (fp_predict - output).pow(2).sum(1).mean()

    return loss

def get_sam_opt_diverse(pretrained_model, model, generator, embedding,old_target_feature,old_target_grad, bn_stats,input_mean,input_std, all_hooks, learnable_var,
                                name_learnable, neighbor_factor=2, loss_type="cos",old_grad=None,temp_factor=0.25, threshold_sim=0):
    #print("CHeck gpu memory after get into function: ")
    #check_gpu_memory()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    softmax = torch.nn.Softmax(dim=1)

    
    data = generator(embedding)
    fp_output = pretrained_model(data)
    q_output = model(data)

    final_block_grad = fp_output - q_output
    if "fpFea" in loss_type:
        raw_feature = torch.reshape(all_hooks[1][-1].inputs, (all_hooks[1][-1].inputs.size()[0], -1))
    else:
        raw_feature = torch.reshape(all_hooks[2][-1].inputs, (all_hooks[2][-1].inputs.size()[0], -1))
    #print("CHeck gpu memory after block 1: ")
    #check_gpu_memory()



    mean_loss = 0
    std_loss = 0
    data_std, data_mean = torch.std_mean(data, [2, 3])
    mean_loss += l2_loss(input_mean, data_mean)
    std_loss += l2_loss(input_std, data_std)

    for (bn_mean, bn_std), hook in zip(bn_stats, all_hooks[0]):
        bn_input = hook.inputs
        data_std, data_mean = torch.std_mean(bn_input, [0, 2, 3])
        mean_loss += l2_loss(bn_mean, data_mean)
        std_loss += l2_loss(bn_std, data_std)
    #torch.cuda.empty_cache()
    #print("Checking memory after get BN loss")
    #check_gpu_memory()

    bn_loss = mean_loss + std_loss


    if "rawfea" in loss_type:
        normalized_feature = raw_feature
    else:
        normalized_feature = torch.nn.functional.normalize(raw_feature)

    if len(old_target_feature) > 0:

        normalized_target_feature = torch.cat([raw_feature,old_target_feature])
        normalized_target_feature = torch.nn.functional.normalize(normalized_target_feature).detach()
    else:
        normalized_target_feature = normalized_feature.detach()




    if len(old_target_grad) > 0:
        target_grad = torch.cat([final_block_grad,old_target_grad])
        target_grad = torch.nn.functional.normalize(target_grad).detach()
    else:
        target_grad = final_block_grad.detach()


    if "normGrad" in loss_type:
        final_block_grad = torch.nn.functional.normalize(final_block_grad)
        target_grad = torch.nn.functional.normalize(target_grad)

    normalized_feature_detach = normalized_feature.detach()+1e-9
    final_block_grad_detach = final_block_grad.detach()+1e-9
    #print("CHeck gpu memory after block 2: ")
    #check_gpu_memory()

    if "maxselfOpt" in loss_type:
        if "maxselfOptUpdateSingle" in loss_type:
            grad_loss = torch.mean(torch.sum(normalized_feature * normalized_feature_detach,dim=1)*torch.sum(final_block_grad * final_block_grad_detach,dim=1))


    with torch.no_grad():
        mask = torch.autograd.grad(grad_loss,embedding,retain_graph=True)[0].detach()
        mask = torch.nn.functional.normalize(mask)*neighbor_factor
    torch.cuda.empty_cache()
    #print("CHeck gpu memory after block 3: ")
    #check_gpu_memory()
    
    new_data = generator(embedding - mask)
    #print("CHeck gpu memory after block 3.5: ")
    #check_gpu_memory()
    new_fp_output = pretrained_model(new_data)
    new_q_output = model(new_data)
    


    new_final_block_grad = new_fp_output - new_q_output
    if "fpFea" in loss_type:
        new_raw_feature = torch.reshape(all_hooks[1][-1].inputs, (all_hooks[1][-1].inputs.size()[0], -1))
    else:
        new_raw_feature = torch.reshape(all_hooks[2][-1].inputs, (all_hooks[2][-1].inputs.size()[0], -1))


    if "rawfea" in loss_type:
        new_normalized_feature = new_raw_feature
    else:
        new_normalized_feature = torch.nn.functional.normalize(new_raw_feature)

    if "normGrad" in loss_type:
        new_normalized_grad = torch.nn.functional.normalize(new_final_block_grad)
    else:
        new_normalized_grad = new_final_block_grad


    normalized_grad = torch.nn.functional.normalize(final_block_grad)
    new_grad_loss = torch.mean(torch.sum(new_normalized_feature*normalized_feature,dim=1) * torch.sum(new_normalized_grad *final_block_grad,dim=1))

    #print("grad_loss  before",new_grad_loss)
    if "normGrad" not in loss_type:
        new_grad_loss = torch.log(1 + torch.exp(-0.1*odd_pow(new_grad_loss + 1e-9, temp_factor)))
    else:
        new_grad_loss = 1-new_grad_loss
    #print("grad_loss  after ",new_grad_loss)
    
    #print("CHeck gpu memory after block 4: ")
    #check_gpu_memory()

    diverse_loss = 0


    if "diverse" in loss_type:

        if "diverseSim" in loss_type:
            print("Using diverseSim")
            diverse_normalized_target_grad = torch.nn.functional.normalize(target_grad)
            diverse_normalized_grad_matrix = torch.matmul(normalized_grad, diverse_normalized_target_grad.T)
            diverse_normalized_fea = torch.nn.functional.normalize(raw_feature)
            diverse_normalized_target_fea = torch.nn.functional.normalize(normalized_target_feature)
            diverse_sim_matrix = torch.matmul(diverse_normalized_fea, diverse_normalized_target_fea.T)
            diverse_sim_matrix.fill_diagonal_(0)
            if "diverseSimTotal" in loss_type:
                print("Using diverseSimTotal")
                diverse_loss = torch.mean(
                    torch.abs(diverse_sim_matrix * diverse_normalized_grad_matrix))
            else:
             #   diverse_loss = torch.mean(torch.abs(torch.max(diverse_sim_matrix * diverse_normalized_grad_matrix, dim=1)[0]))
                final_grad_sim_matrix = torch.clamp(torch.abs(diverse_sim_matrix * diverse_normalized_grad_matrix)-threshold_sim,0, 1e10)
                diverse_loss = torch.mean(torch.max(final_grad_sim_matrix, dim=1)[0])
            print("diverse_loss before ", diverse_loss)
    #print("CHeck gpu memory after block 5: ")
    #check_gpu_memory()

    return new_grad_loss,diverse_loss,bn_loss, raw_feature,final_block_grad, data
def get_sam_opt_diverse_cheating(pretrained_model, model, generator, embedding,old_target_feature,old_target_grad, bn_stats, all_hooks, learnable_var,
                                name_learnable, neighbor_factor=0.5,grad_factor=1, loss_type="cos",old_grad=None,temp_factor=0.25,rand_index=[]):

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    softmax = torch.nn.Softmax(dim=1)
    mismatch_loss = 0
    if loss_type is not "cos":
        data = generator(embedding)
        fp_output = pretrained_model(data)
        q_output = model(data)
        grad_loss = 0

        final_block_grad = fp_output - q_output
        raw_feature = torch.reshape(all_hooks[2][-1].inputs, (all_hooks[2][-1].inputs.size()[0], -1))
        if "rawfea" in loss_type:
            normalized_feature = raw_feature
        else:
            normalized_feature = torch.nn.functional.normalize(raw_feature)


        normalized_target_feature = torch.nn.functional.normalize(old_target_feature).detach()

        target_grad = torch.nn.functional.normalize(old_target_grad).detach()


        sim_matrix = torch.matmul(normalized_feature,normalized_target_feature.T)

        if "normGrad" in loss_type:
            final_block_grad = torch.nn.functional.normalize(final_block_grad)
            target_grad = torch.nn.functional.normalize(target_grad)
        grad_matrix = torch.matmul(final_block_grad, target_grad.T)

        grad_loss += torch.mean(torch.max(sim_matrix * grad_matrix,dim=0)[0])
        print("grad_loss before ", grad_loss)

        if "maxExp" not in loss_type or "maxExpGain" in loss_type:
            if "normGrad" in loss_type:
                grad_loss = 1 - grad_loss
            else:
                grad_loss = torch.log(1 + torch.exp(-grad_factor * odd_pow(grad_loss + 1e-9, temp_factor)))

        print("grad_loss  after", grad_loss)

    diverse_loss = 0
    if "diverse" in loss_type:
        normalized_grad = torch.nn.functional.normalize(final_block_grad)
        if "diverseSim" in loss_type:
            print("Using diverseSim")
            diverse_normalized_target_grad = torch.nn.functional.normalize(target_grad)
            diverse_normalized_grad_matrix = torch.matmul(normalized_grad, diverse_normalized_target_grad.T)
            diverse_normalized_fea = torch.nn.functional.normalize(raw_feature)
            diverse_normalized_target_fea = torch.nn.functional.normalize(normalized_target_feature)
            diverse_sim_matrix = torch.matmul(diverse_normalized_fea, diverse_normalized_target_fea.T)
            diverse_sim_matrix.fill_diagonal_(0)

            diverse_loss = torch.mean(torch.abs(torch.max(diverse_sim_matrix * diverse_normalized_grad_matrix, dim=1)[0]))
            print("diverse_loss before ", diverse_loss)

    return grad_loss,diverse_loss,0
def get_norm(all_data):
    return torch.norm(torch.stack([torch.norm(data) for data in all_data]))

def dot_product(gw_syn, gw_real):
    gw_real_vec = []
    gw_syn_vec = []
    for ig in range(len(gw_real)):
        gw_real_vec.append(gw_real[ig].reshape((-1)))
        gw_syn_vec.append(gw_syn[ig].reshape((-1)))

    gw_real_vec = torch.cat(gw_real_vec, dim=0)

    gw_syn_vec = torch.cat(gw_syn_vec, dim=0)


    dis = torch.matmul(gw_syn_vec,gw_real_vec)
    return dis

def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to('cuda')

    if args == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif 'cos' in args:
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        denominator = torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) +1e-16


        if args == "cosim":
            dis = torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / denominator
        else:
            dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / denominator
        #print("Loss: ",dis)
    elif 'dot' in args:
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)

        dis = -torch.sum(gw_real_vec * gw_syn_vec, dim=-1)


    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis
def odd_pow(input, exponent):
    return input.sign() * input.abs().pow(exponent)
def get_mse_matrix(fea,grad,normalized=False):
    norm_fea = torch.sum(fea**2,dim=1)
    norm_grad = torch.sum(grad ** 2, dim=1)
    norm_combine = norm_fea*norm_grad
    first_matrix = norm_combine[None].repeat(norm_combine.size()[0],1)
    first_matrix = first_matrix + first_matrix.T
    second_matrix = torch.matmul(fea,fea.T)*torch.matmul(grad,grad.T)
    final_mse_matrix = torch.sqrt(first_matrix - 2 * second_matrix)
    if normalized:
        final_mse_matrix = final_mse_matrix/torch.max(final_mse_matrix)
        final_mse_matrix = 1 - final_mse_matrix
    else:
        final_mse_matrix = -final_mse_matrix


    return final_mse_matrix
def get_mse_matrix_fea(fea):
    return -torch.cdist(fea[None],fea[None]).squeeze()

    return final_mse_matrix
def get_gradient_layer(gradient,layer_weight,layer_input,expect_fea,):
    layer_input_unfold = torch.nn.unfold(layer_input)
    mse_loss =  torch.mean(torch.sum(gradient**2,dim=1))
    gradient_layer = torch.autograd.grad(layer_weight,retain_graph=True)[0]

def distill_data_5_grad_matching_genie_2(pretrained_model,model,data,grad_factor, batch_size,total_samples, bn_stats,  iters,neighbor_factor,loss_type,act_quant,bn_factor,diverse_factor,temp_factor,rho,change_iter,extra_data, threshold_sim):
    print("act_quant is ", act_quant)
    lr_z = 0.5
    improve_value=[0,0]



    dataset = []

    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    input_mean = torch.zeros(1, 3).cuda()
    input_std = torch.ones(1, 3).cuda()
    softmax = torch.nn.Softmax(dim=1)
    # print("CHeck gpu memory before change to swing: ")
    # check_gpu_memory()
    hooks = []
    all_hooks = []
    reconstruct_pair = []
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    visited = set()
    teacher_modules = OrderedDict(pretrained_model.named_modules())
    student_modules = OrderedDict(model.named_modules())
    reconstruct_unit = (
        'BasicBlock', 'Bottleneck',  # resnet block
        'ResBottleneckBlock',  # regnet block
        'DwsConvBlock', 'ConvBlock',  # mobilenetb block
        'InvertedResidual',  # mobilenetv2, mnasnet block
        'Linear', 'Conv2d'  # default
    )
    for name, module in teacher_modules.items():
        if (module in reconstruct_unit or module.__class__.__name__ in reconstruct_unit) and module not in visited:
            visited.update(module.modules())
            reconstruct_pair.append((module, student_modules[name], name))

    print("Getting pretrained hooks")
    for name, module in pretrained_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(ActivationHook(module))

    all_hooks_t_block = []
    all_hooks_s_block = []
    for i, (teacher_block, student_block, name) in enumerate(reconstruct_pair):
        t_hook = ActivationHook(teacher_block)
        all_hooks_t_block.append(t_hook)
        print('Adding hook to teacher model at layer ', name)
        if "fc" in name or "classifier.1" in name:
            print('Adding hook to student model at layer ',name)
            print(student_block)
            s_hook = ActivationHook(student_block)
            all_hooks_s_block.append(s_hook)

    all_hooks.append(hooks)
    all_hooks.append(all_hooks_t_block)
    all_hooks.append(all_hooks_s_block)


    if act_quant:
        all_learnable_var = [p for n,p in model.named_parameters() if  p.requires_grad]
    else:
        all_learnable_var = [p for n, p in model.named_parameters() if
                             p.requires_grad and "activation_quantizer" not in n]
    learnable_var = all_learnable_var #all_learnable_var[-2]
    if act_quant:
        name_learnable = [n for n,p in model.named_parameters() if p.requires_grad]
    else:
        name_learnable = [n for n, p in model.named_parameters() if p.requires_grad and "activation_quantizer" not in n]

    grad_ma = None #[None for i in range(len(all_hooks[1]))]
    fea_ma=None
    old_grad = None
    new_sample_2 = None
    all_feature = []
    all_grad = []
    target_feature_val = []
    target_grad_val = []
    curr_selected_rand_index = []

    for i in range(total_samples // batch_size):
        def change_lr(target,new_factor,type="optim"):
            if type == "optim":
                for g in target.param_groups:
                    g['lr'] *= new_factor
            else:
                target.min_lrs = [value*new_factor for value in target.min_lrs]

        z = data[i]["embedding"].requires_grad_() #torch.nn.Parameter(data[i]["embedding"]).requires_grad_().cuda()
        generator = data[i]["generator"]

        opt_z = optim.Adam([z], lr=1e-5)
        scheduler_z = optim.lr_scheduler.ReduceLROnPlateau(opt_z, min_lr=1e-6, verbose=False, patience=100)
        opt_g = optim.Adam(generator.parameters(), lr=1e-4)
        scheduler_g = optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.95)
        z = z.cuda()
        generator= generator.cuda()

        #opt_z = data[i]["opt_z"]
        #scheduler_z = data[i]["scheduler_z"]
        #change_lr(opt_z, 0.1,type="optim")
        #change_lr(scheduler_z, 0.01, type="schedule")

        
        #opt_g = data[i]["opt_g"]
        #scheduler_g = data[i]["scheduler_g"]
        #change_lr(opt_g, 0.1,type="optim")
        # change_lr(scheduler_g, 0.01, type="schedule")
        start=time.time()

        if "selfOpt" in loss_type:
            target_feature = all_feature
            target_grad = all_grad
        #print("Checking memory after load parameter")
        #check_gpu_memory()
        for j in range(iters):

            # initialize the criterion, optimizer, and scheduler

            model.zero_grad()
            opt_z.zero_grad()
            opt_g.zero_grad()

            if "selfOpt" in loss_type:
                if "unChange" in loss_type:
                    if "extraData" in loss_type:
                        curr_grad_match_loss, diverse_loss, bn_loss = get_sam_opt_diverse_cheating(pretrained_model, model, generator, z,
                                                                                     target_feature, target_grad, bn_stats,
                                                                                     all_hooks, learnable_var,
                                                                                     name_learnable,
                                                                                     neighbor_factor=neighbor_factor,
                                                                                     temp_factor=temp_factor,
                                                                                     grad_factor=grad_factor,
                                                                                     loss_type=loss_type, old_grad=old_grad,rand_index = rand_index)
                    else:
                        curr_grad_match_loss, diverse_loss, bn_loss, final_feature, final_grad, final_x = get_sam_opt_diverse(pretrained_model, model, generator, z,
                                                                                     all_feature, all_grad, bn_stats,input_mean,input_std,
                                                                                     all_hooks, learnable_var,
                                                                                     name_learnable,
                                                                                     neighbor_factor=neighbor_factor,
                                                                                     temp_factor=temp_factor,
                                                                                     loss_type=loss_type, old_grad=old_grad,threshold_sim=threshold_sim)
            #torch.cuda.empty_cache()
            #print("Checking memory after get SAM loss")
            #check_gpu_memory()



            total_loss = bn_loss*bn_factor + curr_grad_match_loss*grad_factor + diverse_loss*diverse_factor

            #torch.cuda.empty_cache()
            #print("Checking memory before backward")
            #check_gpu_memory()
            total_loss.backward()

            #torch.cuda.empty_cache()
            #print("Checking memory after backwards")
          #  check_gpu_memory()
            opt_z.step()
            opt_g.step()
            

            scheduler_z.step(total_loss.item())
            #torch.cuda.empty_cache()
            #print("Checking memory after step")
            #check_gpu_memory()


            if (j+1) %1 == 0:
                print(
                    f"Iteration {j}/{iters},lr {opt_z.param_groups[0]['lr']}: grad loss: {curr_grad_match_loss:.2f}, diver loss: {diverse_loss:.2f} ,BN loss: {bn_loss:.2f}, total_loss: {total_loss:.2f}, time: {time.time()-start:.2f}")
                start = time.time()
                scheduler_g.step()
        if len(all_feature) > 0:
            all_feature = torch.cat([all_feature, final_feature])
            all_grad = torch.cat([all_grad, final_grad])
        else:
            all_feature = final_feature
            all_grad = final_grad
        generator=generator.cpu()
        z=z.cpu()




        #torch.cuda.empty_cache()
        #torch.cuda.empty_cache()
        #print("Checking gpu after save grad_ma")
        #check_gpu_memory()


        dataset.append(final_x.detach().clone().cpu())
        print("FInish 1 round ",i)
    for hook in hooks:
        hook.remove()
    for hook in all_hooks_t_block:
        hook.remove()
    for hook in all_hooks_s_block:
        hook.remove()
    dataset = torch.cat(dataset)

    return dataset

