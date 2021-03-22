import argparse, os
import time
import torch
import random
import shutil
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util import AverageMeter
from util import cal_psnr,cal_ssim,cal_ssim_c
from util import save_image_single,save_image_single_c

from model import DKPNet as DKPNet
from custom_transform import SingleTransform
from dataset import CustomDataset
from collections import OrderedDict
import data_generator as dg
from data_generator import DenoisingDataset
import torchvision.models as models
from tensorboardX import SummaryWriter

# Training and Testing settings
parser = argparse.ArgumentParser(description="PyTorch denoise Experiment")
parser.add_argument('--result_dir', dest='result_dir', default='./result', help='Test image result dir')
parser.add_argument('--test_dir', dest='test_dir', default='/media/choi/hdd/xcy/dataset/dataset_denoise/BSD200/', help='Test data dir')
parser.add_argument('--noise_level', type=int, default=25, help='Noise level')
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--color", action="store_true", help="Denoise color image? not necessarily color image, but code specifically for 3-channel image")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--n_type", default=1, type=int, help="noise type")

parser.add_argument("--kp_size", default=3, type=int, help="kernel prediction size(3,5,7), only used for DKPNet")
parser.add_argument("--n1_resblocks", default=9, type=int, help="number of resblocks in shared feature extraction, only used for DKPNet")
parser.add_argument("--n2_resblocks", default=2, type=int, help="number of resblocks in seperated feature extraction, only used for DKPNet")

#tensorboard setting
writer = SummaryWriter('tensorboard/%d'%(time.time()))
best_accuracy = 0
def main():
    global opt, best_accuracy
    opt = parser.parse_args()
    print(opt)

    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    if not os.path.exists(opt.test_dir):
        print("{} not exist".format(opt.test_dir))
        return
    #configure(os.path.join(opt.ckpt_dir, 'log'), flush_secs=5)

    #cuda = False
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    #opt.seed = random.randint(1, 10000)
    opt.seed = 0
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    test_transform = SingleTransform(opt.noise_level,noise_type = opt.n_type)
    test_set = CustomDataset(opt.test_dir, test_transform, opt.color)
    test_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size = 1, shuffle=False)

    print("===> Building model")
    model = DKPNet(kp_size=opt.kp_size, n1=opt.n1_resblocks, n2=opt.n2_resblocks)
    print(model)

    criterion = nn.MSELoss(reduction='sum')
    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        print("Not Using GPU")
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            checkpoint = torch.load(opt.resume)
            
            model_dict = model.state_dict()
            checkpoint_load = {k: v for k, v in (checkpoint['model']).items() if k in model_dict}
            model_dict.update(checkpoint_load)
            model.load_state_dict(model_dict)

            print("=> loading checkpoint '{}'".format(opt.resume))
            opt.start_epoch = 1
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.resume:
        validate(test_data_loader, model, criterion, opt.start_epoch)
        return
    else:
        print("!!!!!!!!!!!!!!!!!! please choose a resume model !!!!!!!!!!")
        return

def validate(test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_psnr = AverageMeter()
    avg_ssim = AverageMeter()

    value_out_path = os.path.join(opt.result_dir, "eval_result.txt")
    F = open(value_out_path,'a')

    model.eval()

    global writer
    with torch.no_grad():
        for i,(image, target,noise_type,noise_level) in enumerate(test_loader):
            image_var = image
            target_var = target

            if opt.cuda:
                image_var = image_var.cuda()
                target_var = target_var.cuda()

            end = time.time()
            clean_image = model(image_var)
            batch_time.update(time.time() - end)
         
            loss = criterion(clean_image, target_var)
            losses.update(loss.item(), image_var.size(0))
            ground_truth = torch.clamp(255 * target_var, 0, 255).byte()
            output_image = torch.clamp(255 * clean_image, 0, 255).byte()
            noise_image = torch.clamp(255 * image_var, 0, 255).byte()
           
            if opt.color == True:
                save_image_single_c(ground_truth.data.cpu().numpy(), noise_image.data.cpu().numpy(), output_image.data.cpu().numpy(),os.path.join(opt.result_dir, 'test%d.png'%i))
                ssim = cal_ssim_c(ground_truth.data.cpu().numpy(), output_image.data.cpu().numpy())
            else:
                save_image_single(ground_truth.data.cpu().numpy(), noise_image.data.cpu().numpy(), output_image.data.cpu().numpy(),os.path.join(opt.result_dir, 'test%d.png'%i))
                ssim = cal_ssim(ground_truth.data.cpu().numpy(), output_image.data.cpu().numpy())

            psnr = cal_psnr(ground_truth.data.cpu().numpy(), output_image.data.cpu().numpy())


            avg_psnr.update(psnr, image_var.size(0))
            avg_ssim.update(ssim, image_var.size(0))

            if i % 1 == 0:
                print('Test Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'Ssim {ssim.val:.4f} ({ssim.avg:.4f})\t'
                  'Psnr {psnr.val:.3f} ({psnr.avg:.3f})'.format(
                   epoch, i, len(test_loader), batch_time=batch_time,
                   ssim=avg_ssim, psnr=avg_psnr))

            F.write('Evaluate Image: [{}/{}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Ssim {ssim.val:.4f} ({ssim.avg:.4f})\t'
              'Psnr {psnr.val:.3f} ({psnr.avg:.3f})\n'.format(
               i, len(test_loader), batch_time=batch_time,
               ssim=avg_ssim, psnr=avg_psnr))

    print("--- Epoch %d  --------- Average PSNR %.3f ---" %(epoch, avg_psnr.avg))

    F.write("Epoch Average: PSNR %.3f ssim %.4f\n"%(avg_psnr.avg,avg_ssim.avg))
    F.close()

    return avg_psnr.avg

if __name__ == "__main__":
    main()
