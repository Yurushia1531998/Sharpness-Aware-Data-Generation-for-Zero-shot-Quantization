import logging
from models import get_model

from distill import distill_data,distill_data_sam,check_gpu_memory,  Generator, SwingConv2d
from utils import get_dataset, evaluate_classifier, evaluate_pretrained_loss, evaluate_sam_loss,ActivationHook,find_parent
import fire
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import time
import copy
import numpy as np
import os
from reconstruct import quantize_model, reconstruct, get_all_optim, enable_training_mode, QuantizableLayer
from reconstruct import reconstruct_final
from reconstruct_origin_genie import reconstruct_origin_Genie, quantize_model_original_Genie
import warnings
import torch.nn.functional as F
import random
import torchvision.transforms as transforms


logging.basicConfig(style='{', format='{asctime} {levelname:8} {name:20} {message}', datefmt='%H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)
def l2_loss(A, B):
    return (A - B).norm()**2 / B.size(0)
def freeze_layer_weight(net):
    for name, param in net.named_parameters():
        if name.endswith(".weight") or name.endswith(".bias"):
            param.requires_grad = False
def freeze_everything(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            param.requires_grad = False
def unfreeze_everything(net):
    for name, param in net.named_parameters():
        if name.endswith("bit_logit") or name.endswith("scale"):
            param.requires_grad = True
def load_warmup_img(model,src):
    all_data = torch.load(src + "/warmup_data.pt")
    warmup_data_slices = all_data.view(*(-1, 128, *all_data.shape[1:]))
    all_label = []
    with torch.no_grad():
        for x in warmup_data_slices:
            if not x.is_cuda:
                x = x.cuda()
            prediction_label = torch.argmax(model(x),1)
            all_label.append(prediction_label)
    warmup_label = torch.cat(all_label)
    print("warmup_label ",warmup_label.unique(return_counts = True))
    return all_data,warmup_label
def load_model(model,weight_path):
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    model.cuda()

def warmup_images_genie(pretrained_model,warmup_iters,bn_stats,total_samples,batch_size,saved_image_folder,loss_warmup_type=None):
    lr_z = 0.01
    lr_g = 0.1

    num_classes = 1000
    latent_dim = 256
    image_dim = (3, 224, 224)
    dataset = []
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    input_mean = torch.zeros(1, 3).cuda()
    input_std = torch.ones(1, 3).cuda()
    hooks = []
    only_data = []
    model = copy.deepcopy(pretrained_model).cuda().eval()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.stride != (1, 1):
                parent = find_parent(model, name)
                setattr(parent, name.split('.')[-1], SwingConv2d(module, jitter_size=1))
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(ActivationHook(module))
    #save_model = {"generator":[],"embedding":[]}
    for i in range(total_samples // batch_size):

        z = torch.randn(batch_size, latent_dim).cuda().requires_grad_()
        opt_z = optim.Adam([z], lr=lr_g)
        generator = Generator(latent_dim=latent_dim).cuda()
        scheduler_z = optim.lr_scheduler.ReduceLROnPlateau(opt_z, min_lr=1e-4, verbose=False, patience=100)
        opt_g = optim.Adam(generator.parameters(), lr=lr_z)
        scheduler_g = optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.95)
        start = time.time()
        for j in range(warmup_iters):

            model.zero_grad()
            opt_z.zero_grad()
            opt_g.zero_grad()


            mean_loss = 0
            std_loss = 0

            x = generator(z)
            output_pretrained = model(x)
            curr_ce_loss = 0

            data_std, data_mean = torch.std_mean(x, [2, 3])
            mean_loss += l2_loss(input_mean, data_mean)
            std_loss += l2_loss(input_std, data_std)
            for (bn_mean, bn_std), hook in zip(bn_stats, hooks):
                bn_input = hook.inputs
                data_std, data_mean = torch.std_mean(bn_input, [0, 2, 3])
                mean_loss += l2_loss(bn_mean, data_mean)
                std_loss += l2_loss(bn_std, data_std)



            total_loss = mean_loss + std_loss
            if loss_warmup_type is not None and  "diverse" in loss_warmup_type:
                total_loss += diverse_loss
            #curr_ce_loss = ce_loss(output_pretrained,curr_label)
            #total_loss += curr_ce_loss


            total_loss.backward()
            opt_z.step()
            opt_g.step()
            scheduler_z.step(total_loss.item())
            if j % 100 == 0:
                print(
                    f"Iteration {j}/{warmup_iters},lr {opt_z.param_groups[0]['lr']}: BN loss: {mean_loss + std_loss:.2f} ,ce_loss: {curr_ce_loss:.2f}, total_loss: {total_loss:.2f}, time: {time.time() - start}")
                scheduler_g.step()
                start = time.time()

           #output_pretrained = model(x)
        generator = generator.cpu()
        z.cpu()
        dataset.append({"embedding":z,"generator":generator})
        #save_model["generator"].append(generator.state_dict())
        #save_model["embedding"].append(z)
        only_data.append(x.detach())
    only_data = torch.cat(only_data)
    if not os.path.isdir(saved_image_folder):
         os.mkdir(saved_image_folder)
    file_saving_name = saved_image_folder + "/warmup_data.pt"
    file_saving_all_datawarmup = saved_image_folder + "/dataset.pt"

    print("Saving ",file_saving_name," ...")
    
    torch.save(only_data,file_saving_name)
    torch.save(dataset, file_saving_all_datawarmup)

    #torch.save(only_data, file_saving_name)
    for hook in hooks:
        hook.remove()
    del model


    return dataset,only_data

def save_img(saved_image_folder,synthetic_sam_data,round=None,num_img=50):
    if not os.path.isdir(saved_image_folder):
        os.mkdir(saved_image_folder)
    if round is not None:
        file_saving_name = saved_image_folder + "/data_round_" + str(round) + ".pt"
    else:
        file_saving_name = saved_image_folder + "/data.pt"
    print("Saving file_saving_name ",file_saving_name," ...")
    torch.save(synthetic_sam_data[:num_img], file_saving_name)
    
def eval(model_name,bit_w,bit_a,weight_path,val_path):
    available_models = (
    'resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet2.0', 'mnasnet1.0', 'mobilenetb', 'resnet20_cifar10','resnet20_cifar100')
    print("available_models")
    print(available_models)
    assert model_name in available_models, f'{model_name} not exist!'
    model = get_model(model_name, pretrained=True).cuda().eval()

    qmodel = quantize_model(
        model, bit_w, bit_a,
        # specify layer to quantize (nn.Identity: residual)
        (nn.Conv2d, nn.Linear, nn.Identity)
    )
    shape = (1, 3, 224, 224)
    rand_data = torch.randn(shape).cuda()
    enable_training_mode(qmodel, False, True, False)
    with torch.no_grad():
        qmodel(rand_data)
    load_model(qmodel, weight_path)
    val_set,all_label = get_dataset(val_path)
    enable_training_mode(qmodel, False, True, False)
    evaluate_classifier(val_set, qmodel)
def main(
    train_path=None,
    val_path="/storage1/tu/imagenet/val",
    model_name='resnet18',
    samples=1024, distill_batch=128, distill_iter=500, lr_g=0.1, lr_z=0.01,
    bit_w=4, bit_a=4,min_quant_sam=None,
    recon_iter=20000, recon_batch=32, round_weight=1.0, iterative_quantize = True,warmup_iters=4000, start_iters=500,
        num_round = 1,grad_matching=True,using_genie=True,grad_factor=1 ,save_img_epoch=10,neighbor_factor=2,
        loss_type=None, saved_image_folder="",saved_warmup_image_folder=None, bn_factor=1,diverse_factor=1,temp_factor=0.2,rho=0.1,change_iter=None, weight_path=None,loss_warmup_type=None,enable_act_train=False,cut_off_rate=2, warmup_quantize_model_path=None,
        threshold_sim=0, use_wandb=False
):
    """Quantize the model with synthetic dataset.

    Args:
        train_path: Training set path. (If set, use training data rather than distilled images.)
        val_path: Validation set path.
        model_name: model architecture
        samples: # of distilled images to generate (or # of sampled training set)
        distill_batch: batch size at distillation
        distill_iter: # of iterations per distillation batch.
        lr_g: lr of generator
        lr_z: lr of latent vector
        bit_w: weight bitwidth
        bit_a: activation bitwidth
        recon_iter: # of iterations per reconstruction
        recon_batch: batch size at reconstruction
        round_weight: rounding loss weight for quantization
    """
    if use_wandb:
        import wandb
        wandb.init(project='Rebuttal_Dung', name=f'W{bit_w}A{bit_a}trainset_{samples}_iter_{distill_iter}_diver{diverse_factor}_BN{bn_factor}_grad{grad_factor}_neighbor{neighbor_factor}_threshold{threshold_sim}')
        wandb.run.log_code(".")
    
    
    if weight_path is not None:
        eval(model_name,bit_w,bit_a,weight_path,val_path)
        return
    random_number = random.randint(0,10)
    print("###############random_number########################",random_number)
    saved_image_folder +=f"{model_name}_{bit_w}A{bit_a}trainset_{samples}_iter_{distill_iter}_diver{diverse_factor}_BN{bn_factor}_grad{grad_factor}_neighbor{neighbor_factor}_threshold{threshold_sim}_time{random_number}"

    if not os.path.isdir(saved_image_folder):
        os.mkdir(saved_image_folder)
    available_models = ('resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet2.0', 'mnasnet1.0', 'mobilenetb')
    assert model_name in available_models, f'{model_name} not exist!'
    model = get_model(model_name, pretrained=True)
    
    model.require_grad=False
    model.eval()
    model=model.cuda()

    qmodel = quantize_model(
        model, bit_w, bit_a,
        # specify layer to quantize (nn.Identity: residual)
        (nn.Conv2d, nn.Linear, nn.Identity)
    )
    # print("CHeck gpu memory before get val set: ")
    # check_gpu_memory()


    #print(qmodel.named_modules())
    from utils import get_dataset, evaluate_classifier
    perclass_var = samples if "perclass" in loss_type else 0
    if "resnet20" in model_name:
        val_set, _ = get_dataset(val_path)
    else:
        val_set,all_label = get_dataset(val_path,perclass=perclass_var)
    print("Size of val set: ",len(val_set))
    test_init = next(iter(torch.utils.data.DataLoader(
    val_set, batch_size=1, num_workers=4)))[0].cuda()
    
    
    #freeze_layer_weight(model)
    #freeze_layer_weight(qmodel)
    freeze_everything(model)
    freeze_everything(qmodel)
        
    hooks, bn_stats = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(ActivationHook(module))
            bn_stats.append((module.running_mean.detach().clone().cuda(),
                                torch.sqrt(module.running_var + 1e-6).detach().clone().cuda()))
    for hook in hooks:
        hook.remove()
    #print("CHeck gpu memory before wamup: ")
    #check_gpu_memory()
    print("saved_warmup_image_folder", saved_warmup_image_folder)
    if saved_warmup_image_folder is None:
        print("Warming up images...")
        if using_genie:
            print("using genie")
            all_warmup_data,warmup_data = warmup_images_genie(model, warmup_iters, bn_stats, samples, distill_batch,
                                                            saved_image_folder,loss_warmup_type)
            print("len wamrup data", len(all_warmup_data))
        else:
            warmup_data,warmup_label = warmup_images(model, warmup_iters, bn_stats,samples,distill_batch,saved_image_folder )
    else:
        print("Loading warmup images...")
        def load_dataset(path):
            dataset = torch.load(path)
            all_data = []
            num_group = len(dataset)
            print(num_group)
            with torch.no_grad():
                for group_index in range(num_group):
                    group = dataset[group_index]

                    generator = group['generator'].cuda()
                    #generator = generator
                    embedding = group['embedding'].cuda()
                    #embedding = embedding
                    all_data.append(generator(embedding).detach())
                all_data = torch.cat(all_data)
            return dataset,all_data
        all_warmup_data, warmup_data = load_dataset(saved_warmup_image_folder)
      #  print("number of images", len(warmup_data))
      #  "/storage2/tu/meta_ptq/256_128_warmup_model/dataset.pt"
        #print("all_data shape", warmup_data.shape)
    print("Finish warming up images...")
    
    #     print("Loading warmup images...")ửam
    
    data_for_warmup = warmup_data
    print("Warming up model...")
    if warmup_quantize_model_path is None:
        unfreeze_everything(qmodel)
        print("init warmup model")
        data_for_warmup = data_for_warmup.cuda()
        reconstruct(
                model, qmodel, data_for_warmup,
                # specify layer to reconstruct
                (
                    'BasicBlock', 'Bottleneck',  # resnet block
                    'ResBottleneckBlock',  # regnet block
                    'DwsConvBlock', 'ConvBlock',  # mobilenetb block
                    'InvertedResidual',  # mobilenetv2, mnasnet block
                    'Linear', 'Conv2d'  # default
                ),
                round_weight=round_weight, iterations=warmup_iters, batch_size=recon_batch
            )
        torch.save(qmodel.state_dict(), f'{saved_image_folder}/Warmup_{model_name}_{bit_w}w{bit_a}.pth')
    else:
        print("Loading warmup model...")
        for name, module in qmodel.named_modules():
            if isinstance(module, QuantizableLayer):
                module.enable_act_quant = True
        with torch.no_grad():
            _ = qmodel(test_init)
        qmodel.load_state_dict(torch.load(warmup_quantize_model_path))
                
    print("FInished Warming up model...")
    if bit_a is None:
        act_quant = False
    else:
        act_quant = True

    print("act_quant is 0 ",act_quant)
    if enable_act_train:
        enable_training_mode(qmodel, (False,True), act_quant,True)
    else:
        enable_training_mode(qmodel, False, act_quant, True)
    print("Start the loop of quantizing model")
    print("CHeck gpu memory at the start of the loop: ")

  #  check_gpu_memory()
    for i in range(num_round):

        if i == 0:
            distill_data_iters = start_iters
        else:
            distill_data_iters = distill_iter

        if grad_matching:
            all_optim_scale = None

            extra_data = None
            freeze_everything(qmodel)
            train_set = distill_data_sam(model, qmodel, data=all_warmup_data, grad_factor=grad_factor,
                                                        batch_size=distill_batch, total_samples=samples,
                                                        bn_stats=bn_stats,
                                                        iters=distill_data_iters,neighbor_factor = neighbor_factor,loss_type=loss_type,
                                                            act_quant = act_quant,bn_factor=bn_factor,diverse_factor=diverse_factor,temp_factor= temp_factor,rho = rho,change_iter= change_iter,extra_data = extra_data,
                                                            threshold_sim=threshold_sim
                                                        )
            unfreeze_everything(qmodel)

            #train_set = sam_dataset.data
            if i % save_img_epoch == 0:     
                file_saving_name = saved_image_folder + "/data_round_"+str(i)+".pt"
                save_img(file_saving_name, train_set,round = i)

        torch.cuda.empty_cache()
        enable_training_mode(qmodel, True, True,False)
        name_save_train_set = f"W{bit_w}A{bit_a}trainset_{samples}_iter_{distill_iter}_diver{diverse_factor}_BN{bn_factor}_grad{grad_factor}_neighbor{neighbor_factor}_threshold{threshold_sim}.pt"
        path_save_trainset = saved_image_folder + "/" + name_save_train_set
        torch.save(train_set, path_save_trainset)
        
        # quantized model 
        qmodel2 = quantize_model_original_Genie(
            model, bit_w, bit_a,
            # specify layer to quantize (nn.Identity: residual)
            (nn.Conv2d, nn.Linear, nn.Identity)
        )
      #  enable_training_mode(qmodel2, True, True,False)
        train_set = train_set.cuda()
        reconstruct_origin_Genie(
            model, qmodel2, train_set,
            # specify layer to reconstruct
            (
                'BasicBlock','Bottleneck', # resnet block
                'ResBottleneckBlock', # regnet block
                'DwsConvBlock', 'ConvBlock', # mobilenetb block
                'InvertedResidual', # mobilenetv2, mnasnet block
                'Linear', 'Conv2d' # default
            ),
            round_weight=round_weight, iterations=recon_iter, batch_size=recon_batch
        )
        if i%save_img_epoch == 0:
            print("Saving model of epoch ",i,"....")
            torch.save(qmodel2.state_dict(), saved_image_folder + "/model.pt")
        torch.cuda.empty_cache()
        print("performance after the ", i, " round:")

       # enable_training_mode(qmodel2, False,True,False)
        acc = evaluate_classifier(val_set, qmodel2)
        if use_wandb:
            import wandb
            wandb.log({"accuracy":acc})

        #evaluate_sam_loss(qmodel,model,boundary_set)
    print("Saving final model....")
    torch.save(qmodel2.state_dict(), saved_image_folder+"/quantized_model.pt")



def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False
    seed_all(1029)
    fire.Fire(main)