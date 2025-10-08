import logging
import copy
import torch
from torch import nn, Tensor
from collections import OrderedDict
from utils import ActivationHook, find_parent
from quantizer import WeightQuantizer, ActivationQuantizer
from pynvml import *
nvmlInit()
log = logging.getLogger(__name__)


def test_freezed_model(net):
    for name, param in net.named_parameters():
        if (name.endswith(".weight") or name.endswith(".bias")) and param.requires_grad == True:
            print("Param ",name," is still true")
def test_require_grad(net):
    for name, param in net.named_parameters():
        if (name.endswith(".weight") or name.endswith(".bias")) and param.requires_grad == True:
            print("Param ",name," require grad")
def test_require_grad_all(net):
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            print("Param ",name," require grad")

class QuantizableLayer(nn.Module):
    """
    Wrapper module that performs fake quantization operations.
    """

    def __init__(self, org_module, weight_bits, activation_bits):
        super(QuantizableLayer, self).__init__()

        self.org_module = org_module
        self.enable_act_quant = False
        self.weight_quantizer = None
        self.activation_quantizer = None

        if weight_bits is not None:
            if hasattr(org_module, 'weight'):
                self.weight = org_module.weight
                self.weight_quantizer = WeightQuantizer(org_module.weight, weight_bits)
                self.org_module._parameters.pop('weight', None)

        if activation_bits is not None:
            self.activation_quantizer = ActivationQuantizer(activation_bits)
    def update_params(self):
        self.org_module.weight = self.weight_quantizer()
    def forward(self, x: Tensor):
        if self.activation_quantizer and self.enable_act_quant:
            # trick to share same quantization parameter between residual and conv
            if hasattr(x, 'tensor_quantizer'):
                assert self.activation_quantizer.n_bits <= x.tensor_quantizer.n_bits, 'Activation bitwidth becomes smaller.'
                self.activation_quantizer = x.tensor_quantizer
            else:
                x.tensor_quantizer = self.activation_quantizer
            x = self.activation_quantizer(x)

        if self.weight_quantizer:
            #print("Using weight quantizer")
            #print("Original weight:")
            #print(self.weight)
            self.org_module.weight = self.weight_quantizer()
            #print("New weight:")
            #print(self.org_module.weight)
            #print("----")
        '''
        print("org_module:")
        print(self.org_module)
        print("weight_quantizer:")
        print(self.weight_quantizer)
        print("activation_quantizer:")
        print(self.activation_quantizer)
        print("enable_act_quant:")
        print(self.enable_act_quant)
        '''
        x = self.org_module(x)

        return x

class LinearTempDecay:
    def __init__(self, iter_max, rel_start_decay, start_t, end_t):
        self.t_max = iter_max
        self.start_decay = rel_start_decay * iter_max
        self.start_b = start_t
        self.end_b = end_t

    def __call__(self, cur_iter):
        if cur_iter < self.start_decay:
            return self.start_b
        else:
            rel_t = (cur_iter-self.start_decay) / (self.t_max-self.start_decay)
            return self.end_b + (self.start_b-self.end_b)*max(0.0, 1 - rel_t)


def reconstruct_block(
    block, x, x_q, y,curr_optim_scale=None,curr_optim_bit=None,curr_scheduler=None,curr_decay=None,curr_iteration=0,
    iterations=20000, round_weight=1.0,
    lr_w_scale=0.0001, lr_a_scale=0.00004, lr_bit=0.001,
    annealing_range=(20,2), annealing_warmup=0.2, batch_size=32
):

    param_a_scale = []
    param_w_scale = []
    param_bit = []


    if curr_optim_scale is not None:
        opt_scale = curr_optim_scale
        opt_bit = curr_optim_bit
        scheduler_scale = curr_scheduler
        temp_decay = curr_decay
        iters = curr_iteration
    else:
        for module in block.modules():

            if isinstance(module, WeightQuantizer):
                module.train_mode = True
                param_w_scale.append(module.scale)
                param_bit.append(module.bit_logit)

            elif isinstance(module, ActivationQuantizer):
                module.train_mode = True
                param_a_scale.append(module.scale)

        opt_scale = torch.optim.Adam([
            {"params": param_w_scale, 'lr': lr_w_scale},
            {"params": param_a_scale, 'lr': lr_a_scale}
        ])


        opt_bit = torch.optim.Adam(param_bit, lr=lr_bit)

        scheduler_scale = torch.optim.lr_scheduler.CosineAnnealingLR(opt_scale, T_max=iterations)


        temp_decay = LinearTempDecay(
            iterations, rel_start_decay=annealing_warmup,
            start_t=annealing_range[0], end_t=annealing_range[1])


        iters = 0

    final_iters = iters + iterations
    while iters < final_iters:
        perms = torch.randperm(len(x)).view(batch_size, -1)
        for idx in perms:
            iters += 1

            x_mix = torch.where(torch.rand_like(x[idx]) < 0.5, x_q[idx], x[idx]) # use QDrop
            y_gt = y[idx]
            if not x_mix.is_cuda:
                x_mix = x_mix.cuda()
            if not y_gt.is_cuda:
                y_gt = y_gt.cuda()
            y_q = block(x_mix)

            recon_loss = (y_q - y_gt).pow(2).sum(1).mean()
            round_loss = 0

            annealing_temp = temp_decay(iters)
            if iters >= annealing_warmup*iterations:
                for module in block.modules():
                    if isinstance(module, WeightQuantizer):
                        round_loss += (1 - (2*module.soft_target() - 1).abs().pow(annealing_temp)).sum()

            total_loss = recon_loss + round_loss * round_weight

            opt_scale.zero_grad()
            opt_bit.zero_grad()
            total_loss.backward()
            opt_scale.step()
            opt_bit.step()
            scheduler_scale.step()

            if iters == 1 or iters % 1000 == 0:
                log.info(
                    f'{iters}/{iterations}, Total loss: {total_loss:.3f} (rec:{recon_loss:.3f}, round:{round_loss:.3f})'
                    +f'\tb={annealing_temp:.2f}\tcount={iters}')

            if iters >= iterations:
                break
    # Finish optimization, use hard rounding.
    for module in block.modules():
        if isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = False




def reconstruct_block_2(
    block,all_optim_set, x, x_q, y,
    iterations=20000, round_weight=1.0,
    lr_w_scale=0.0001, lr_a_scale=0.00004, lr_bit=0.001,
    annealing_range=(20,2), annealing_warmup=0.2, batch_size=32
):




    temp_decay = LinearTempDecay(
        iterations, rel_start_decay=annealing_warmup,
        start_t=annealing_range[0], end_t=annealing_range[1])

    iters = 0
    while iters < iterations:
        perms = torch.randperm(len(x)).view(batch_size, -1)
        for idx in perms:
            iters += 1

            x_mix = torch.where(torch.rand_like(x[idx]) < 0.5, x_q[idx], x[idx]) # use QDrop
            y_q = block(x_mix)

            recon_loss = (y_q - y[idx]).pow(2).sum(1).mean()
            round_loss = 0

            annealing_temp = temp_decay(iters)
            if iters >= annealing_warmup*iterations:
                for module in block.modules():
                    if isinstance(module, WeightQuantizer):
                        round_loss += (1 - (2*module.soft_target() - 1).abs().pow(annealing_temp)).sum()

            total_loss = recon_loss + round_loss * round_weight

            all_optim_set[0].zero_grad()
            all_optim_set[1].zero_grad()
            total_loss.backward()
            all_optim_set[0].step()
            all_optim_set[1].step()
            all_optim_set[2].step()

            if iters == 1 or iters % 1000 == 0:
                log.info(
                    f'{iters}/{iterations}, Total loss: {total_loss:.3f} (rec:{recon_loss:.3f}, round:{round_loss:.3f})'
                    +f'\tb={annealing_temp:.2f}\tcount={iters}')

            if iters >= iterations:
                break

    # Finish optimization, use hard rounding.
    for module in block.modules():
        if isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = False


def quantize_model(model, bit_w,bit_w_second_layer, bit_a, quant_ops):
    """Convert a full precision model to a quantized model.

    Args:
        model: full precision model
        bit_w: weight bitwidth
        bit_a: activation bitwidth
        quant_ops (tuple): List of operator types to quantize

    Returns:
        nn.Module: quantized model
    """
    qmodel = copy.deepcopy(model)
    qconfigs = []
    for name, module in qmodel.named_modules():
        if isinstance(module, quant_ops):
            qconfigs.append({'name': name, 'module': module, 'bit_w': bit_w, 'bit_a': bit_a})
            parent = find_parent(qmodel, name)

    # keep first and last layer to 8bit
    qconfigs[0] = {**qconfigs[0], 'bit_w': 8, 'bit_a': None} # Input image is already quantized.
    # if you want to use QDrop quantization setting, comment out the code below
    if bit_w_second_layer == 8:
        qconfigs[1] = {**qconfigs[1], 'bit_a': bit_w_second_layer} # BRECQ keeps the second layer’s input to 8bit
    qconfigs[-1] = {**qconfigs[-1], 'bit_w': 8, 'bit_a': 8}

    for qconfig in qconfigs:
        parent = find_parent(qmodel, qconfig['name'])
        setattr(parent, qconfig['name'].split('.')[-1], 
                QuantizableLayer(qconfig['module'], qconfig['bit_w'], qconfig['bit_a']))

    return qmodel

def get_all_optim(teacher, student,reconstruct_unit):
    original_lr_w_scale = 0.0001
    original_lr_a_scale = 0.00004
    original_lr_bit = 0.001

    lr_w_scale = 0.001
    lr_a_scale = 0.0004
    lr_bit = 0.01
    iterations = 20000

    teacher_modules = OrderedDict(teacher.named_modules())
    student_modules = OrderedDict(student.named_modules())
    reconstruct_pair = []
    all_optim_dict ={}

    visited = set()
    for name, module in teacher_modules.items():
        if (module in reconstruct_unit or module.__class__.__name__ in reconstruct_unit) and module not in visited:
            visited.update(module.modules())
            reconstruct_pair.append((module, student_modules[name], name))
    param_a_scale = []
    param_w_scale = []
    param_bit = []

    all_optim_scale = []
    all_optim_bit = []
    all_scheduler = []
    all_decay = []
    annealing_range = (20, 2)
    annealing_warmup = 0.2
    for i, (teacher_block, student_block, name) in enumerate(reconstruct_pair):
        log.info(f'Getting optim ({i}/{len(reconstruct_pair)}): {name}')
        curr_block_param_a_scale = []
        curr_block_param_w_scale = []
        curr_block_param_bit = []

        for module in student_block.modules():
            if isinstance(module, WeightQuantizer):
                module.train_mode = True
                param_w_scale.append(module.scale)
                param_bit.append(module.bit_logit)
                curr_block_param_w_scale.append(module.scale)
                curr_block_param_bit.append(module.bit_logit)

            elif isinstance(module, ActivationQuantizer):
                module.train_mode = True
                param_a_scale.append(module.scale)
                curr_block_param_a_scale.append(module.scale)

        curr_opt_scale = torch.optim.Adam([
            {"params": curr_block_param_w_scale, 'lr': original_lr_w_scale},
            {"params": curr_block_param_a_scale, 'lr': original_lr_a_scale}
        ])
        all_optim_scale.append(curr_opt_scale)
        all_optim_bit.append(torch.optim.Adam([
            {"params": curr_block_param_bit, 'lr': original_lr_bit}
        ]))
        all_scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(curr_opt_scale, T_max=iterations))
        all_decay.append(LinearTempDecay(
            iterations, rel_start_decay=annealing_warmup,
         start_t=annealing_range[0], end_t=annealing_range[1]))


    opt_all = torch.optim.Adam([
            {"params": param_w_scale, 'lr': lr_w_scale},
            {"params": param_a_scale, 'lr': lr_a_scale},
            {"params": param_bit, 'lr': lr_bit}
        ])

    scheduler_scale = torch.optim.lr_scheduler.CosineAnnealingLR(opt_all, T_max=iterations)
    return opt_all,scheduler_scale, all_optim_scale,all_optim_bit, all_scheduler, all_decay

def reconstruct(teacher, student, cali_data, reconstruct_unit,all_optim_scale = None,all_optim_bit =None,all_scheduler=None,all_decay=None,curr_iteration=0, **kwargs):
    """Reconstructs the quantized model.

    Args:
        teacher: full precision model
        student: quantized model to reconstruct
        cali_data (tensor): calibration dataset
        reconstruct_unit (tuple): A list of block or layer to reconstruct
    """

    teacher_modules = OrderedDict(teacher.named_modules())
    student_modules = OrderedDict(student.named_modules())
    reconstruct_pair = []
    #print("Test require grad student")
    #test_require_grad_all(student)
    #print("Test require grad teacher")
    #test_require_grad_all(teacher)
    visited = set()
    for name, module in teacher_modules.items():
        if (module in reconstruct_unit or module.__class__.__name__ in reconstruct_unit) and module not in visited:
            visited.update(module.modules())
            reconstruct_pair.append((module, student_modules[name], name))



    batch_size = 32
    #print("Memory updated before getting activation")
    #check_gpu_memory()
    for i, (teacher_block, student_block, name) in enumerate(reconstruct_pair):
        log.info(f'Recontruct ({i}/{len(reconstruct_pair)}): {name}')
        for name, module in student_block.named_modules():
            if isinstance(module, QuantizableLayer):
                module.enable_act_quant = True
            elif isinstance(module, (WeightQuantizer, ActivationQuantizer)):
                module.train_mode = True

        act_x, act_y, act_x_q = [], [], []

        cali_data_slices = cali_data.view(*(-1, batch_size, *cali_data.shape[1:]))

        t_hook = ActivationHook(teacher_block)
        s_hook = ActivationHook(student_block)
        '''
        for x in cali_data_slices:
            teacher(x)
            student(x)
            act_x.append(t_hook.inputs)
            act_y.append(t_hook.outputs)
            act_x_q.append(s_hook.inputs)
        '''


        with torch.no_grad():
            for x in cali_data_slices:
                if not x.is_cuda:
                    x = x.cuda()
                teacher(x)
                student(x)
                act_x.append(t_hook.inputs.cpu())
                act_y.append(t_hook.outputs.cpu())
                act_x_q.append(s_hook.inputs.cpu())
            #print("Memory updated after getting activation")
            #check_gpu_memory()

            act_x = torch.cat(act_x)
            act_y = torch.cat(act_y)
            act_x_q = torch.cat(act_x_q)

        t_hook.remove()
        s_hook.remove()
        if all_optim_scale is not None:
            reconstruct_block(student_block, act_x, act_x_q, act_y,all_optim_scale[i],all_optim_bit[i],all_scheduler[i],all_decay[i],curr_iteration, **kwargs)
        else:
            reconstruct_block(student_block, act_x, act_x_q, act_y, **kwargs)
    enable_training_mode(student,False,True,False)


def reconstruct_genie(teacher, student, cali_data, reconstruct_unit, **kwargs):
    """Reconstructs the quantized model.

    Args:
        teacher: full precision model
        student: quantized model to reconstruct
        cali_data (tensor): calibration dataset
        reconstruct_unit (tuple): A list of block or layer to reconstruct
    """
    teacher_modules = OrderedDict(teacher.named_modules())
    student_modules = OrderedDict(student.named_modules())
    reconstruct_pair = []

    visited = set()
    for name, module in teacher_modules.items():
        if (module in reconstruct_unit or module.__class__.__name__ in reconstruct_unit) and module not in visited:
            visited.update(module.modules())
            reconstruct_pair.append((module, student_modules[name], name))

    for i, (teacher_block, student_block, name) in enumerate(reconstruct_pair):
        log.info(f'Recontruct ({i}/{len(reconstruct_pair)}): {name}')
        for name, module in student_block.named_modules():
            if isinstance(module, QuantizableLayer):
                module.enable_act_quant = True
            elif isinstance(module, (WeightQuantizer, ActivationQuantizer)):
                module.train_mode = True

        act_x, act_y, act_x_q = [], [], []
        batch_size = 32
        cali_data_slices = cali_data.view(*(-1, batch_size, *cali_data.shape[1:]))

        t_hook = ActivationHook(teacher_block)
        s_hook = ActivationHook(student_block)

        with torch.no_grad():
            for x in cali_data_slices:
                teacher(x)
                student(x)
                act_x.append(t_hook.inputs)
                act_y.append(t_hook.outputs)
                act_x_q.append(s_hook.inputs)
        act_x = torch.cat(act_x)
        act_y = torch.cat(act_y)
        act_x_q = torch.cat(act_x_q)

        t_hook.remove()
        s_hook.remove()
        reconstruct_block(student_block, act_x, act_x_q, act_y, **kwargs)

    for name, module in student.named_modules():
        if isinstance(module, QuantizableLayer):
            module.enable_act_quant = True
        elif isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = False

def check_gpu_memory():
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')
    print("memory allocated: ",torch.cuda.memory_allocated())
    print("max memory allocate: ", torch.cuda.max_memory_allocated())
    print(" ")
def enable_training_mode(model,enable_train_weight=True,enable_act = True,enable_meta=False):
    for name, module in model.named_modules():
        if isinstance(module, QuantizableLayer):
            module.enable_act_quant = enable_act
        elif isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            if isinstance(enable_train_weight,tuple):

                if isinstance(module, WeightQuantizer):
                    module.train_mode = enable_train_weight[0]
                else:
                    module.train_mode = enable_train_weight[1]
            else:
                module.train_mode = enable_train_weight
            module.meta_mode= enable_meta
def enable_training_mode_block(block,enable=True):
    # Finish optimization, use hard rounding.
    for module in block.modules():
        if isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = enable
def meta_step_block_wise():
    pass
def meta_step_all_block():
    pass
def reconstruct_2(iters,teacher, student, cali_data,meta_opt_scale, reconstruct_unit, **kwargs):
    """Reconstructs the quantized model.

    Args:
        teacher: full precision model
        student: quantized model to reconstruct
        cali_data (tensor): calibration dataset
        reconstruct_unit (tuple): A list of block or layer to reconstruct
    """



    #print("Memory updated at step 0 ")
    #check_gpu_memory()

    test_require_grad(student)
    teacher_modules = OrderedDict(teacher.named_modules())
    student_modules = OrderedDict(student.named_modules())
    reconstruct_pair = []
    #print("Memory updated at step 1 ")
    #check_gpu_memory()

    visited = set()
    for name, module in teacher_modules.items():
        if (module in reconstruct_unit or module.__class__.__name__ in reconstruct_unit) and module not in visited:
            visited.update(module.modules())
            reconstruct_pair.append((module, student_modules[name], name))
    ##print("Memory updated at step 2 ")
    ##check_gpu_memory()

    annealing_range = (20, 2)
    annealing_warmup = 0.2
    round_weight = 1.0
    iterations = 500
    temp_decay = LinearTempDecay(
        iterations, rel_start_decay=annealing_warmup,
        start_t=annealing_range[0], end_t=annealing_range[1])
    #print("Memory updated at step 3 ")
    #check_gpu_memory()

    for name, module in student.named_modules():
        if isinstance(module, QuantizableLayer):
            module.enable_act_quant = True
        elif isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = True
    act_y, act_x_mix,all_hook_t,all_hook_s = {}, {}, {}, {}
    for i, (teacher_block, student_block, name) in enumerate(reconstruct_pair):
        all_hook_t[name] = ActivationHook(teacher_block)
        all_hook_s[name] = ActivationHook(student_block)
    #batch_size = 32
    #cali_data_slices = cali_data.view(*(-1, batch_size, *cali_data.shape[1:]))

    #print("Memory updated at step 3.5 ")
    #check_gpu_memory()
    with torch.no_grad():
        teacher(cali_data)
    #print("Memory updated at step 3.6 ")
    #check_gpu_memory()
    student(cali_data)
    ##print("Memory updated at step 3.7 ")
    ##check_gpu_memory()
    for i, name in enumerate(all_hook_t.keys()):
        #print("Name is ",name)
        #print("Memory updated at step 3.8 ",i)
        #check_gpu_memory()
        act_y[name] = all_hook_t[name].outputs

        act_x_mix[name] = torch.where(torch.rand_like(all_hook_t[name].inputs) < 0.5, all_hook_s[name].inputs,
                                all_hook_t[name].inputs)
        all_hook_t[name].remove()
        all_hook_s[name].remove()
    ##print("Memory updated at step 3.9 ")
    ##check_gpu_memory()
    total_loss = 0
    for i, (teacher_block, student_block, name) in enumerate(reconstruct_pair):



        #log.info(f'Recontruct ({i}/{len(reconstruct_pair)}): {name}')
        ##print("Memory updated at step 4 of round ",i)
        ##check_gpu_memory()







        round_loss = 0

        #print("Memory updated at step 8.5 of round ",i)
        #check_gpu_memory()
        annealing_temp = temp_decay(iters)
        for module in student_block.modules():
            if isinstance(module, WeightQuantizer):
                round_loss += (1 - (2 * module.soft_target() - 1).abs().pow(annealing_temp)).sum()

        #print("Memory updated at step 9 of round ",i)
        #check_gpu_memory()

        total_loss += ((act_y[name] - student_block(act_x_mix[name])).pow(2).sum(1).mean() + round_loss * round_weight)

        #print("Memory updated at the end of round ",i)
        #check_gpu_memory()
        #print("--------------")
        #meta_opt_scale.step(total_loss)


    #meta_opt_scale.zero_grad()
    #meta_opt_bit.zero_grad()
    #total_loss.backward()
    #print("Memory updated before finishing reconstruc ")
    #check_gpu_memory()


    meta_opt_scale.step(total_loss)
    '''
    del total_loss
    all_key = list(act_x_mix.keys())
    for i, name in enumerate(all_key):
        act_x_mix[name].detach()
        act_y[name].detach()
    #print("Memory updated after update scale ")
    '''
    ##check_gpu_memory()
    #meta_opt_bit.step()
    #meta_scheduler_scale.step()

    '''
    if iters == 1 or iters % 1000 == 0:
            log.info(
                f'{iters}/{iterations}, Total loss: {total_loss:.3f} (rec:{recon_loss:.3f}, round:{round_loss:.3f})'
                + f'\tb={annealing_temp:.2f}\tcount={iters}')
    

    for name, module in student.named_modules():
        if isinstance(module, QuantizableLayer):
            module.enable_act_quant = True
        elif isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = False
    '''
