import logging
from models import get_model
from distill import distill_data
from utils import get_dataset, evaluate_classifier
import fire
import torch
from torch import nn
import os
from reconstruct import quantize_model, reconstruct, QuantizableLayer
from quantizer import ActivationQuantizer,WeightQuantizer


logging.basicConfig(style='{', format='{asctime} {levelname:8} {name:20} {message}', datefmt='%H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)
def save_img(saved_image_folder,synthetic_sam_data,round=None):
    if not os.path.isdir(saved_image_folder):
        os.mkdir(saved_image_folder)
    if round is not None:
        file_saving_name = saved_image_folder + "/data_round_" + str(round) + ".pt"
    else:
        file_saving_name = saved_image_folder + "/data.pt"
    print("Saving file_saving_name ...")
    torch.save(synthetic_sam_data, file_saving_name)

def enable_training_mode(model,enable=True,enable_act = True,enable_meta=False):
    for name, module in model.named_modules():
        if isinstance(module, QuantizableLayer):
            module.enable_act_quant = enable_act
        elif isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = enable
            module.meta_mode= enable_meta
def load_model(model,weight_path):
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    model.cuda()
def main(
    train_path=None,
    val_path=None,
    model_name='resnet18',
    samples=1024, distill_batch=128, distill_iter=4000, lr_g=0.1, lr_z=0.01,
    bit_w=4, bit_a=4,
    recon_iter=20000, recon_batch=32, round_weight=1.0,weight_path=None,iterative_quantize = False
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
    available_models = ('resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet2.0', 'mnasnet1.0', 'mobilenetb')
    assert model_name in available_models, f'{model_name} not exist!'
    model = get_model(model_name, pretrained=True).cuda().eval()


    qmodel = quantize_model(
    	model, bit_w, bit_a,
    	# specify layer to quantize (nn.Identity: residual)
    	(nn.Conv2d, nn.Linear, nn.Identity)
    )


    
    shape = (1, 3, 224, 224)
    rand_data = torch.randn(shape).cuda()
    enable_training_mode(qmodel,False,True,False)
    with torch.no_grad():
        qmodel(rand_data)
    load_model(qmodel,weight_path)


    '''
    reconstruct(
        model, qmodel, train_set,
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
    '''

    val_set = get_dataset(val_path)
    enable_training_mode(qmodel,False,True,False)
    evaluate_classifier(val_set, qmodel)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    fire.Fire(main)