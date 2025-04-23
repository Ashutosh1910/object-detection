from data import config,voc0712,voc,VOCDetection,detection_collate
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
HOME = os.path.expanduser("~")
VOC_ROOT = os.path.join(HOME, "data/VOCdevkit/")
class Config:
    def __init__(self):
       self.cuda=False
args=Config()
args.cuda = torch.cuda.is_available() and args.cuda
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_device('cuda')
else:
    torch.set_default_device('cpu')

if not os.path.exists('weights/'):
    os.mkdir('weights/')

# torch.set_default_device('mps')
def train():
    MEANS = (104, 117, 123)
    dataset = VOCDetection(root=VOC_ROOT,
                               transform=SSDAugmentation(300,
                                                         MEANS))


    ssd_net = build_ssd('train', 300, 21)
    net = ssd_net
    net
    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True


    vgg_weights = torch.load('weights/' + 'vgg16_reducedfc.pth')
    print('loading network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()



    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9,
                          weight_decay=5e-4) 
    criterion = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('loading dataset')
    step_index = 0
    data_loader = data.DataLoader(dataset, 32,
                                  num_workers=4,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(0, 120000):


        # load train data
        images, targets = next(batch_iterator)
        # targets= [target.to(torch.device('mps')) for target in targets]

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        # print(loss.item())

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')


            
    torch.save(ssd_net.state_dict(),
            'weights/' + '' + 'VOC' + '.pth')





if __name__ == '__main__':
    train()
