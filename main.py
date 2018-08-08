import torchvision
import torch
import torch.nn.functional as fn
import logging
from torch.autograd import Variable
logger = logging.getLogger(__name__)

import argparse
parser = argparse.ArgumentParser(description = 'A Pytorch Version of CapsNet')
parser.add_argument('--batch-size',type=int,default=128,metavar='BS')
parser.add_argument('--total-epochs',type=int,default=5000,metavar='TE')
parser.add_argument('--learn-rate',type=float,default = 1e-4,metavar='LR')
parser.add_argument('--caps-num',type=int,default=32,metavar='CN')
parser.add_argument('--caps-dim',type=int,default=8,metavar='CD')
parser.add_argument('--out-dim',type=int,default=16,metavar='OD')
parser.add_argument('--cuda',type=bool,default=True)
args = parser.parse_args()
def squash(data):
    sqrmod = data.pow(2).sum(2)
    mod = sqrmod.sqrt()
    data = (sqrmod / (1+sqrmod) / mod).unsqueeze(2) * data
    return data

def to_one_hot(ids):
    ids = ids.view(-1,1)
    out_tensor = torch.FloatTensor(ids.shape[0],10)
    out_tensor.zero_()
    out_tensor.scatter_(dim=1, index=ids.data, value=1.)
    out_tensor = Variable(out_tensor)
    if args.cuda:
        out_tensor = out_tensor.cuda()
    return out_tensor

class PrimaryCapsules(torch.nn.Module):
    def __init__(self):
        super(PrimaryCapsules,self).__init__()
        self.conv_1 = torch.nn.Conv2d(
                in_channels = 1,
                out_channels = args.caps_num*args.caps_dim,
                kernel_size = 9,
                stride = 1
                )
        self.conv_2 = torch.nn.Conv2d(
                in_channels = args.caps_num*args.caps_dim,
                out_channels = args.caps_num*args.caps_dim,
                kernel_size = 9,
                stride = 2
                )

    def forward(self,data):
        batch_size,in_channels,H,W = data.shape
        data = self.conv_1(data)
        data = self.conv_2(data)
        data = data.view(batch_size,args.caps_num,args.caps_dim,6,6).permute(0,1,3,4,2).contiguous().view(batch_size,args.caps_num,6*6*args.caps_dim)
        data = squash(data)
        return data

class ArgreementRouting(torch.nn.Module):
    def __init__(self,
            in_caps_num = 32,
            out_caps_num = 10,
            in_caps_dim = args.caps_dim*6*6,
            out_caps_dim = args.out_dim
            ):
        super(ArgreementRouting,self).__init__()
        self.W = torch.nn.Parameter(torch.Tensor(32,args.caps_dim*6*6,10*args.out_dim))
        self.b = torch.nn.Parameter(torch.zeros(in_caps_num,out_caps_num))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / 32
        self.W.data.uniform_(-stdv,stdv)

    def forward(self,caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.W).view(-1,32,10,16)
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = fn.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        b_batch = self.b.expand((batch_size, input_caps, output_caps))
        for r in range(3):
            v = v.unsqueeze(1)
            b_batch = b_batch + (u_predict * v).sum(-1)

            c = fn.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
            s = (c * u_predict).sum(dim=1)
            v = squash(s)
        return v


class CapsNet(torch.nn.Module):
    def __init__(self):
        super(CapsNet,self).__init__()
        self.primary_capsules = PrimaryCapsules()
        self.argreement_routing = ArgreementRouting()

    def forward(self,data):
        data = self.primary_capsules(data)
        data = self.argreement_routing(data)
        return data.pow(2).sum(-1).sqrt(),data

class CapsNetWithReconstruction(torch.nn.Module):
    def __init__(self):
        super(CapsNetWithReconstruction,self).__init__()
        self.capsnet = CapsNet()
        self.dense_1 = torch.nn.Linear(args.out_dim*10, 512)
        self.dense_2 = torch.nn.Linear(512,1024)
        self.dense_3 = torch.nn.Linear(1024,784)

    def forward(self,data,target):
        scores, result = self.capsnet(data)
        if target is None:
            return scores,result,None
        data = result
        mask = Variable(torch.zeros(data.size()[0], 10), requires_grad=False)
        if args.cuda:
            mask = mask.cuda()
        target = target.long()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        data = data*mask
        data = data.view(-1, args.out_dim * 10)
        data = torch.nn.functional.relu(self.dense_1(data))
        data = torch.nn.functional.relu(self.dense_2(data))
        data = torch.nn.functional.sigmoid(self.dense_3(data))
        return scores,result,data

if __name__ == '__main__':
    import tqdm

    trainset = torchvision.datasets.MNIST('mnist',download=True,train=True, transform = torchvision.transforms.ToTensor())
    validset = torchvision.datasets.MNIST('mnist',download=True,train=False,transform = torchvision.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset=trainset,batch_size = args.batch_size,shuffle=True)
    validloader = torch.utils.data.DataLoader(dataset=validset,batch_size = args.batch_size,shuffle=True)

    model = CapsNetWithReconstruction()
    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr = args.learn_rate)
    for epoch in range(args.total_epochs):
        logger.info("Epoch %d"%epoch)
        for idx,(labels,targets) in enumerate(tqdm.tqdm(trainloader)):
            labels,_targets = Variable(labels),Variable(targets)
            targets = to_one_hot(_targets)
            if args.cuda:
                labels,targets = labels.cuda(),targets.cuda()
                _targets = _targets.cuda()

            preds,_,reconstruct = model(labels,_targets)
            loss = (targets*torch.nn.functional.relu(.9-preds).pow(2) + .5 * (1-targets)*torch.nn.functional.relu(preds-.1).pow(2)).mean() + .001*(labels.view(-1,784)-reconstruct).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total = 0
        total_correct = 0
        for idx,(labels,targets) in enumerate(validloader):
            labels,targets = Variable(labels),Variable(targets)
            if args.cuda:
                labels,targets = labels.cuda(),targets.cuda()

            preds_onehot,_,_ = model(labels,None)
            _,preds  = torch.max(preds_onehot,dim=1)
            total += len(preds)
            total_correct += (preds == targets).sum().data.cpu().numpy()[0]

        print(total,total_correct)
