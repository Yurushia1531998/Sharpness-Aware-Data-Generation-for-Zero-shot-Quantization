import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def __init__(self):
        super(MetaModule, self).__init__()

        self.org_module = None
        self.enable_act_quant = False
        self.weight_quantizer = None
        self.activation_quantizer = None


        '''
        if weight_bits is not None:
            if hasattr(self.org_module, 'weight'):
                self.weight_quantizer = WeightQuantizer(self.org_module.weight, weight_bits)
                self.org_module._parameters.pop('weight', None)

        if activation_bits is not None:
            self.activation_quantizer = ActivationQuantizer(activation_bits)
        '''

    def get_optim_parameter(self):
        param_a_scale = []
        param_w_scale = []
        param_bit = []

        for module in block.modules():
            if isinstance(module, WeightQuantizer):
                module.train_mode = True
                param_w_scale.append(module.scale)
                param_bit.append(module.bit_logit)

            elif isinstance(module, ActivationQuantizer):
                module.train_mode = True
                param_a_scale.append(module.scale)

        self.opt_scale = torch.optim.Adam([
            {"params": param_w_scale, 'lr': lr_w_scale},
            {"params": param_a_scale, 'lr': lr_a_scale}
        ])

        self.opt_bit = torch.optim.Adam(param_bit, lr=lr_bit)
        self.scheduler_scale = torch.optim.lr_scheduler.CosineAnnealingLR(opt_scale, T_max=iterations)

        self.temp_decay = LinearTempDecay(
            iterations, rel_start_decay=annealing_warmup,
            start_t=annealing_range[0], end_t=annealing_range[1])
    def get_optim_parameter_per_block(self):
        param_a_scale = []
        param_w_scale = []
        param_bit = []
        if not hasattr(self.reconstruct_pair):
            self.get_reconstruction_layer_list()
        for i, (teacher_block, student_block, name) in enumerate(self.reconstruct_pair):
            for module in block.modules():
                if isinstance(module, WeightQuantizer):
                    module.train_mode = True
                    param_w_scale.append(module.scale)
                    param_bit.append(module.bit_logit)

                elif isinstance(module, ActivationQuantizer):
                    module.train_mode = True
                    param_a_scale.append(module.scale)

            self.opt_scale[name] = torch.optim.Adam([
                {"params": param_w_scale, 'lr': lr_w_scale},
                {"params": param_a_scale, 'lr': lr_a_scale}
            ])

            self.opt_bit[name] = torch.optim.Adam(param_bit, lr=lr_bit)
            self.scheduler_scale[name] = torch.optim.lr_scheduler.CosineAnnealingLR(opt_scale, T_max=iterations)

            self.temp_decay[name] = LinearTempDecay(
                iterations, rel_start_decay=annealing_warmup,
                start_t=annealing_range[0], end_t=annealing_range[1])


    def params(self):
       for name, param in self.named_params(self):
            yield param
    
    def named_leaves(self):
        return []
    
    def named_submodules(self):
        return []
    
    def named_params(self, curr_module=None, memo=None, prefix=''):       
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
                    
        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p
    
    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)
    def get_reconstruction_layer_list(self,teacher,student,reconstruct_unit):
        teacher_modules = OrderedDict(teacher.named_modules())
        student_modules = OrderedDict(student.named_modules())
        self.reconstruct_pair = []

        visited = set()
        for name, module in teacher_modules.items():
            if (module in reconstruct_unit or module.__class__.__name__ in reconstruct_unit) and module not in visited:
                visited.update(module.modules())
                self.reconstruct_pair.append((module, student_modules[name], name))

    def reconstruct_block(self,name,
            block, x, x_q, y,
            iters = 0, round_weight=1.0,
            lr_w_scale=0.0001, lr_a_scale=0.00004, lr_bit=0.001,
            annealing_range=(20, 2), annealing_warmup=0.2, batch_size=32
    ):
        iters = 0
        x_mix = torch.where(torch.rand_like(x) < 0.5, x_q, x)  # use QDrop
        y_q = block(x_mix)

        recon_loss = (y_q - y).pow(2).sum(1).mean()
        round_loss = 0

        annealing_temp = temp_decay(iters)
        if iters >= annealing_warmup * iterations:
            for module in block.modules():
                if isinstance(module, WeightQuantizer):
                    round_loss += (1 - (2 * module.soft_target() - 1).abs().pow(annealing_temp)).sum()

        total_loss = recon_loss + round_loss * round_weight

        self.opt_scale[name].zero_grad()
        self.opt_bit[name].zero_grad()
        total_loss.backward()
        opt_scale.step()
        opt_bit.step()
        self.scheduler_scale[name].step()

        if iters == 1 or iters % 1000 == 0:
            log.info(
                f'{iters}/{iterations}, Total loss: {total_loss:.3f} (rec:{recon_loss:.3f}, round:{round_loss:.3f})'
                + f'\tb={annealing_temp:.2f}\tcount={iters}')

        # Finish optimization, use hard rounding.
        for module in block.modules():
            if isinstance(module, (WeightQuantizer, ActivationQuantizer)):
                module.train_mode = False
    def update_params_quant_block_wise(self,x,lr_inner, first_order=False, weight_params=None, act_params=None, detach=False):
        for i, (teacher_block, student_block, name) in enumerate(self.reconstruct_pair):
            log.info(f'Recontruct ({i}/{len(self.reconstruct_pair)}): {name}')
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
            self.reconstruct_block(name,student_block, act_x, act_x_q, act_y, **kwargs)

    def get_grad_quant(self):
        iters = 0
        while iters < iterations:
            perms = torch.randperm(len(x)).view(batch_size, -1)
            for idx in perms:
                iters += 1

                x_mix = torch.where(torch.rand_like(x[idx]) < 0.5, x_q[idx], x[idx])  # use QDrop
                y_q = block(x_mix)

                recon_loss = (y_q - y[idx]).pow(2).sum(1).mean()
                round_loss = 0

                annealing_temp = self.temp_decay(iters)
                if iters >= annealing_warmup * iterations:
                    for module in block.modules():
                        if isinstance(module, WeightQuantizer):
                            round_loss += (1 - (2 * module.soft_target() - 1).abs().pow(annealing_temp)).sum()

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
                        + f'\tb={annealing_temp:.2f}\tcount={iters}')

                if iters >= iterations:
                    break

        # Finish optimization, use hard rounding.
        for module in block.modules():
            if isinstance(module, (WeightQuantizer, ActivationQuantizer)):
                module.train_mode = False

    def update_params_quant(self, lr_inner, first_order=False, weight_params=None,act_params=None, detach=False):
        if source_params is not None:
            for name_scaled_w_quant,scale_w_quant,name_scaled_act_quant, scale_act_quant, src in zip(self.named_weight_quant(self),self.named_act_quant(self), weight_params):

                grad_scaled_w = src
                if first_order:
                    grad = to_var(grad.detach().data)
                new_scale_w_quant = scale_w_quant - lr_inner * grad_scaled_w
                self.set_param(self, name_scaled_w_quant, new_scale_w_quant)

            for name_scaled_act_quant, scale_act_quant, src in zip(self.named_act_quant(self), act_params):

                grad_scaled_act = src
                if first_order:
                    grad = to_var(grad.detach().data)
                new_scale_act_quant = scale_act_quant - lr_inner * grad_scaled_act
                self.set_param(self, name_scaled_act_quant, new_scale_act_quant)
        else:

            print("Not found grad input!!!! Program is breaking!!")
            print("---------------------------------------------")



    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    def update_params_sam_meta_model(self,grad_norm,source_params):
        for tgt, src in zip(self.named_params(self), source_params):
            scale = self.rho / (grad_norm + 1e-12)

            name_t, param_t = tgt
            # name_s, param_s = src
            # grad = param_s.grad
            # name_s, param_s = src
            grad = src
            if first_order:
                e_w = (torch.pow(param, 2) if self.adaptive else 1.0) * grad * scale.to(param)
            tmp = param_t - lr_inner * e_w
            self.set_param(self, name_t, tmp)




    def set_param(self,curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)
            
    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())   
                
    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
       
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
       
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)
        
        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:           
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

        
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)
            
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class LeNet(nn.Module):
    def __init__(self, n_out):
        super(LeNet, self).__init__()
    
        layers = []
        layers.append(MetaConv2d(1, 6, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(MetaConv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        
        layers.append(MetaConv2d(16, 120, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        
        self.main = nn.Sequential(*layers)
        
        layers = []
        layers.append(MetaLinear(120, 84))
        layers.append(nn.ReLU(inplace=True))
        layers.append(MetaLinear(84, n_out))
        
        self.fc_layers = nn.Sequential(*layers)


    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120)
        return self.fc_layers(x).squeeze()


# class LeNet(MetaModule):
#     def __init__(self, n_out):
#         super(LeNet, self).__init__()
    
#         layers = []
#         layers.append(nn.Conv2d(1, 6, kernel_size=5))
#         layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

#         layers.append(nn.Conv2d(6, 16, kernel_size=5))
#         layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        
#         layers.append(nn.Conv2d(16, 120, kernel_size=5))
#         layers.append(nn.ReLU(inplace=True))
        
#         self.main = nn.Sequential(*layers)
        
#         layers = []
#         layers.append(nn.Linear(120, 84))
#         layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Linear(84, n_out))
        
#         self.fc_layers = nn.Sequential(*layers)
        
#     def forward(self, x):
#         x = self.main(x)
#         x = x.view(-1, 120)
#         return self.fc_layers(x).squeeze()