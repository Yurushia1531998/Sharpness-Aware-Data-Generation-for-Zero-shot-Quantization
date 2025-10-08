import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
def find_parent(module, name: str):
    """Recursively apply getattr and returns parent of module"""
    if name == '':
        raise ValueError('Cannot Found')
    for sub_name in name.split('.')[: -1]:
        if hasattr(module, sub_name):
            module = getattr(module, sub_name)
        else:
            raise ValueError('submodule name not exist')
    return module

class ActivationHook():
    """
    Forward_hook used to get the output of the intermediate layer. 
    """

    def __init__(self, module,detach=False):
        super(ActivationHook, self).__init__()
        self.inputs, self.outputs = None, None
        self.handle = module.register_forward_hook(self.hook)
        self.detach = detach

    def hook(self, module, input, output):

        #def tensor_hook(grad):
        #    self.grad = grad
        if self.detach:
            self.inputs = input[0].detach()  # arg tuple
            self.outputs = output.detach()
        else:
            self.inputs = input[0]  # arg tuple
            self.outputs = output
        #self.inputs.register_hook(tensor_hook)

    def remove(self):
        self.handle.remove()

class BackwardHook():
    """
    Forward_hook used to get the output of the intermediate layer.
    """

    def __init__(self, module):
        super(BackwardHook, self).__init__()
        self.grad_inputs, self.grad_outputs = None, None
        self.handle = module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_input, grad_output):
        self.grad_inputs = grad_input  # arg tuple
        self.grad_outputs = grad_output

    def remove(self):
        self.handle.remove()
def getRandomData(dataset='cifar10', batch_size=512, for_inception=False):
    """
    get random sample dataloader
    dataset: name of the dataset
    batch_size: the batch size of random data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    if dataset == 'cifar10':
        size = (3, 32, 32)
        num_data = 10000
    elif dataset == 'imagenet':
        num_data = 10000
        if not for_inception:
            size = (3, 224, 224)
        else:
            size = (3, 299, 299)
    else:
        raise NotImplementedError
    dataset = UniformDataset(length=10000, size=size, transform=None)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=32)
    return data_loader

def get_dataset(data_path, num_samples=None,get_matrix=False,perclass=0):
    data_transform = transforms.Compose([
        transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_path, data_transform)
    all_label = None

    if num_samples is not None:
        # sample random subset from train dataset
        size_perm = len(dataset)
        if size_perm < num_samples:
            subset_indexes = torch.randperm(size_perm)
        else:
            subset_indexes = torch.randperm(size_perm)[:num_samples]
        if perclass>0:
            subset_indexes = [i for i in range(len(dataset)) if dataset.imgs[i][1] in range(128)]
        all_label = torch.tensor([dataset.imgs[index][1] for index in subset_indexes])
        dataset = torch.utils.data.Subset(dataset, subset_indexes)

    elif perclass > 0:
        subset_indexes = [i for i in range(len(dataset)) if dataset.imgs[i][1] in range(perclass)]
        all_label = torch.tensor([dataset.imgs[index][1] for index in subset_indexes])
        dataset = torch.utils.data.Subset(dataset, subset_indexes)
    if get_matrix:
        all_data_matrix = []
        #all_data_label = []
        dataloader = DataLoader(dataset, batch_size=32, drop_last=False)
        for data,label in iter(dataloader):
            all_data_matrix.append(data)
            #all_data_label.append(label)
        all_data_matrix = torch.stack(all_data_matrix,0)
        all_data_matrix_shape = all_data_matrix.size()
        all_data_matrix = all_data_matrix.reshape(-1,all_data_matrix_shape[-3],all_data_matrix_shape[-2],all_data_matrix_shape[-1])
        #all_data_label = torch.stack(all_data_label,0)
        return dataset,all_data_matrix
    else:
        return dataset,all_label
def get_dataset_random(data_path, num_samples=None):
    data_transform = transforms.Compose([
        transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_path, data_transform)

    if num_samples is not None:
        # sample random subset from train dataset
        subset_indexes = torch.randperm(len(dataset))[:num_samples]
        subset = torch.utils.data.Subset(dataset, subset_indexes)

    return subset

@torch.no_grad()
def evaluate_classifier(dataset, model, batch_size=64, workers=4, print_freq=50):
    device = next(model.parameters()).device
    model.to(device).eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    total, correct = 0, 0
    for i, (images, target) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        pred = model(images)
        correct += int((pred.argmax(dim=1)==target).sum())
        total += images.shape[0]

        if i % print_freq == 0:
            print(f"Test {i}/{len(data_loader)}: {correct/total*100:.2f}")

    print(f"Test: {correct/total*100:.2f}")
    return correct/total


def evaluate_sam_loss( model,pretrained_model,train_set, batch_size=64, workers=4, print_freq=50):

    unfreeze_layer_weight(model)
    device = next(model.parameters()).device
    model.to(device).eval()
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)



    total, correct = 0, 0

    #ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    for i, images in enumerate(data_loader):
        images = images.to(device)

        pred = model(images)
        correct += int((pred.argmax(dim=1)==target).sum())
        total += images.shape[0]

        y_ptrained = pretrained_model(images)


        total_loss += ce_loss(F.log_softmax(pred, dim=1), F.log_softmax(y_ptrained, dim=1))

    grad_value = torch.autograd.grad(total_loss,model.named_parameters())
    grad_dict = {n:grad_value[i] for i,(n,p) in enumerate(model.named_parameters()) if p.require_grad}
    grad_value = [grad for grad in grad_value if grad is not None]
    with torch.no_grad():
        sam_manager = SAM(grad_dict,grad_value,model)
        sam_manager.get_sam_model()
        for i, (images, target) in enumerate(data_loader):
            images = images.to(device)

            pretrained_output = pretrained_model(images)
            kl_sam_loss = kl_loss(F.log_softmax(model(images), dim=1),F.log_softmax(pretrained_output, dim=1))
        sam_manager.reset_model()

    print(f"Test samloss: {kl_sam_loss:.2f}")


def evaluate_pretrained_loss( model,train_set,bn_stats, batch_size=64, workers=4, print_freq=50):
    print("train_set ")
    print(train_set)
    device = next(model.parameters()).device
    model.to(device).eval()
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(ActivationHook(module))

    total = 0
    mean_loss, std_loss = 0, 0
    input_mean = torch.zeros(1, 3).cuda()
    input_std = torch.ones(1, 3).cuda()
    ce_loss = nn.CrossEntropyLoss()
    for i, (images, target) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        pred = model(images)
        total += images.shape[0]





        data_std, data_mean = torch.std_mean(images, [2, 3])
        mean_loss += l2_loss(input_mean, data_mean)
        std_loss += l2_loss(input_std, data_std)

        for (bn_mean, bn_std), hook in zip(bn_stats, hooks):
            bn_input = hook.inputs
            data_std, data_mean = torch.std_mean(bn_input, [0, 2, 3])
            mean_loss += l2_loss(bn_mean, data_mean)
            std_loss += l2_loss(bn_std, data_std)


        total_ce_loss += ce_loss(pred, target)
    mean_loss /= total
    std_loss /= total
    total_ce_loss /= total



    print(f"Test all pretrained loss: mean_loss {mean_loss:.2f}, std_loss: {std_loss:.2f}, ce_loss: {total_ce_loss:.2f}")



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis