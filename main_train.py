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
from util import cal_psnr,cal_ssim
from util import save_image,save_image_single

from model import DKPNet as DKPNet
from torch.optim.lr_scheduler import MultiStepLR
from custom_transform import SingleRandomCropTransform
from custom_transform import SingleTransform
from dataset import CustomDataset
from collections import OrderedDict
import data_generator as dg
from data_generator import DenoisingDataset
import torchvision.models as models
from tensorboardX import SummaryWriter

# Training and Testing settings
parser = argparse.ArgumentParser(description="PyTorch denoise Experiment")
parser.add_argument('--ck', dest='ckpt_dir', default='./checkpoint', help="Models are saved here")
parser.add_argument('--result_dir', dest='result_dir', default='./result', help='Test image result dir')
parser.add_argument('--train_dir', dest='train_dir',default='/home/xcy/dataset/dataset_denoise/BSD500/', help="Train data dir")
parser.add_argument('--test_dir', dest='test_dir', default='/home/xcy/dataset/dataset_denoise/BSD68_gray/', help='Test data dir')
parser.add_argument('--noise_level', type=int, default=25, help='Noise level')
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")

parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument("--snr", default=0.95, type=float, help="dropout param setting")
parser.add_argument("--n_type", default=1, type=int, help="noise type")

parser.add_argument("--lamda", default=1, type=float, help="dropout param setting")

# Parameters for DKPNet structure
parser.add_argument("--kp_size", default=3, type=int, help="kernel prediction size(3,5,7), only used for DKPNet")
parser.add_argument("--n1_resblocks", default=9, type=int, help="number of resblocks in shared feature extraction, only used for DKPNet")
parser.add_argument("--n2_resblocks", default=2, type=int, help="number of resblocks in seperated feature extraction, only used for DKPNet")
parser.add_argument("--n_feats", default=64, type=int, help="number of feature channels, only used for DKPNet")

#tensorboard setting
writer = SummaryWriter('tensorboard/%d'%(time.time()))
best_accuracy = 0
def main():
    global opt, best_accuracy
    opt = parser.parse_args()
    print(opt)

    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    if not os.path.exists(opt.train_dir):
        print("{} not exist".format(opt.train_dir))
        return
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
    test_set = CustomDataset(opt.test_dir, test_transform)
    test_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size = 1, shuffle=False)
    print("===> Building model")
    model = DKPNet(kp_size=opt.kp_size, n1=opt.n1_resblocks, n2=opt.n2_resblocks, n_feats=opt.n_feats)
    print(model)

    criterion = nn.MSELoss(reduction='sum')
    print("===> Setting GPU")
    if cuda:
        #model = torch.nn.DataParallel(model).cuda()
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
            opt.start_epoch = checkpoint["epoch"] + 1
            print("=> start_epoch set to '{}'".format(opt.start_epoch))
            #opt.start_epoch = 1
            best_accuracy = checkpoint['best_accuracy']
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
   
    print("===> Training")
    step_1 = int(opt.nEpochs*3/10)
    step_2 = int(opt.nEpochs*6/10)
    step_3 = int(opt.nEpochs*9/10)
    print(opt.snr)
    scheduler = MultiStepLR(optimizer,milestones=[step_1,step_2,step_3],gamma=0.25)
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        #Load training data
        xs = dg.datagenerator(data_dir=opt.train_dir)
        xs = xs.astype('float32')/255.0
        xs = torch.from_numpy(xs.transpose((0,3,1,2)))
        #xs = torch.from_numpy(xs)
        train_set = DenoisingDataset(xs,opt.noise_level)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, drop_last=True,shuffle=True)
    
        scheduler.step(epoch)
        print("===>Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
        train(training_data_loader, optimizer, model, criterion, epoch)

        is_best = 0
        if epoch%2 == 1:
            psnr = validate(test_data_loader, model, criterion, epoch)
            is_best = psnr > best_accuracy
            best_accuracy = max(psnr, best_accuracy)
            save_checkpoint({'epoch': epoch,
                         'best_accuracy':best_accuracy,
                         'model': model.state_dict()}, is_best, epoch)

def train(training_data_loader, optimizer, model, criterion, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_psnr = AverageMeter()
    global Writer
    model.train()
    end = time.time()

    value_out_path = os.path.join(opt.ckpt_dir, "train_log.txt")
    F = open(value_out_path,'a')

    for i, batch in enumerate(training_data_loader, 1):
        data_time.update(time.time() - end)
        noise_image, groundtruth= batch[0],batch[1]
        if opt.cuda:
            noise_image = noise_image.cuda()
            groundtruth = groundtruth.cuda()

        noise_image.requires_grad_()
        groundtruth.requires_grad_(False)
        clean_image = model(noise_image)
        
        loss = criterion(clean_image, groundtruth)/(noise_image.size()[0]*2)
        
        losses.update(loss.item(), clean_image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ground_truth = torch.clamp(255 * groundtruth, 0, 255).byte()
        output_clean_image = torch.clamp(255 * clean_image, 0, 255).byte()
        psnr = cal_psnr(ground_truth.data.cpu().numpy(), output_clean_image.data.cpu().numpy())
        avg_psnr.update(psnr, noise_image.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Psnr {psnr.val:.3f} ({psnr.avg:.3f})'.format(
                   epoch, i, len(training_data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, psnr=avg_psnr))
            F.write('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Psnr {psnr.val:.3f} ({psnr.avg:.3f})\n'.format(
                   epoch, i, len(training_data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, psnr=avg_psnr))
    
    F.close()
    writer.add_scalar("train_loss",losses.avg,epoch)
    writer.add_scalar('train_avg_psnr', avg_psnr.avg, epoch)

def validate(test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_psnr = AverageMeter()
    avg_ssim = AverageMeter()

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
           
            save_image_single(ground_truth.data.cpu().numpy(), noise_image.data.cpu().numpy(), output_image.data.cpu().numpy(),os.path.join(opt.result_dir, 'test%d.png'%i))
            psnr = cal_psnr(ground_truth.data.cpu().numpy(), output_image.data.cpu().numpy())
            ssim = cal_ssim(ground_truth.data.cpu().numpy(), output_image.data.cpu().numpy())

            avg_psnr.update(psnr, image_var.size(0))
            avg_ssim.update(ssim, image_var.size(0))

            if i % 1 == 0:
                print('Test Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'Ssim {ssim.val:.4f} ({ssim.avg:.4f})\t'
                  'Psnr {psnr.val:.3f} ({psnr.avg:.3f})'.format(
                   epoch, i, len(test_loader), batch_time=batch_time,
                   ssim=avg_ssim, psnr=avg_psnr))


    print("--- Epoch %d  --------- Average PSNR %.3f ---" %(epoch, avg_psnr.avg))

    value_out_path = os.path.join(opt.ckpt_dir, "eval_log.txt")
    F = open(value_out_path,'a')
    F.write("Epoch %d: PSNR %.3f ssim %.4f\n"%(epoch,avg_psnr.avg,avg_ssim.avg))
    F.close()

    return avg_psnr.avg

def save_checkpoint(state, is_best, epoch):
    model_out_path = os.path.join(opt.ckpt_dir, "model_epoch_{}.pth".format(epoch))
    torch.save(state, model_out_path)
    #print("Checkpoint saved to {}".format(model_out_path))
    if is_best:
        best_model_name = os.path.join(opt.ckpt_dir, "model_best.pth")
        shutil.copyfile(model_out_path, best_model_name)
        print('Best model {} saved to {}'.format(model_out_path, best_model_name))

if __name__ == "__main__":
    main()
