import torchvision
import torch
import logging
logger = logging.getLogger(__name__)

def squash(data):
    sqrmod = data.pow(2).sum(2)
    mod = sqrmod.sqrt()
    data = (sqrmod / (1+sqrmod) / mod).unsqueeze(2) * data
    return data

def to_one_hot(ids):
    ids = torch.LongTensor(ids).view(-1,1)
    out_tensor = torch.FloatTensor(ids.shape[0],10)
    out_tensor.zero_()
    out_tensor.scatter_(dim=1, index=ids, value=1.)
    return out_tensor

class PrimaryCapsules(torch.nn.Module):
    def __init__(self):
        super(PrimaryCapsules,self).__init__()
        self.conv_1 = torch.nn.Conv2d(
                in_channels = 1,
                out_channels = 256,
                kernel_size = 9,
                stride = 1
                )
        self.conv_2 = torch.nn.Conv2d(
                in_channels = 256,
                out_channels = 32*8,
                kernel_size = 9,
                stride = 2
                )

    def forward(self,data):
        batch_size,in_channels,H,W = data.shape
        data = self.conv_1(data)
        data = self.conv_2(data)
        data = data.view(batch_size,32,8,6,6).permute(0,1,3,4,2).contiguous().view(batch_size,32,6*6*8)
        data = squash(data)
        return data

class ArgreementRouting(torch.nn.Module):
    def __init__(self):
        super(ArgreementRouting,self).__init__()
        self.W = torch.nn.Parameter(torch.Tensor(32,8*6*6,10*16))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / 32
        self.W.data.uniform_(-stdv,stdv)

    def forward(self,data):
        batch_size,in_caps_num,in_dims = data.shape
        out_caps_num = 10
        out_dims = 16
        W = self.W.unsqueeze(0).expand(batch_size,-1,-1,-1)
        data = data.unsqueeze(2)
        data = torch.matmul(data,W)
        u_hat = data.view(batch_size,in_caps_num,out_caps_num,out_dims)

        b = torch.zeros(in_caps_num,out_caps_num)
        for it in range(3):
            c = torch.nn.functional.softmax(b,dim=0)
            c = c.unsqueeze(0).unsqueeze(3).expand(batch_size,-1,-1,out_dims)
            data = (u_hat*c).sum(1)
            v = data
            a = u_hat*v.unsqueeze(1).expand(-1,in_caps_num,-1,-1)
            a = a.pow(2).sum(-1).sqrt().mean(0)
            b = b + a;
        return data


class CapsNet(torch.nn.Module):
    def __init__(self):
        super(CapsNet,self).__init__()
        self.primary_capsules = PrimaryCapsules()
        self.argreement_routing = ArgreementRouting()

    def forward(self,data):
        data = self.primary_capsules(data)
        data = self.argreement_routing(data)
        return data.pow(2).sum(-1).sqrt(),data


if __name__ == '__main__':
    import argparse
    import tqdm
    parser = argparse.ArgumentParser(description = 'A Pytorch Version of CapsNet')
    parser.add_argument('--batch-size',type=int,default=100,metavar='BS')
    parser.add_argument('--total-epochs',type=int,default=10,metavar='TE')
    parser.add_argument('--learn-rate',type=float,default = 1e-2,metavar='LR')
    args = parser.parse_args()

    trainset = torchvision.datasets.MNIST('mnist',download=True,train=True, transform = torchvision.transforms.ToTensor())
    validset = torchvision.datasets.MNIST('mnist',download=True,train=False,transform = torchvision.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset=trainset,batch_size = args.batch_size)
    validloader = torch.utils.data.DataLoader(dataset=validset,batch_size = args.batch_size)

    model = CapsNet()
    optimizer = torch.optim.Adam(model.parameters(),lr = args.learn_rate)
    for epoch in range(args.total_epochs):
        logger.info("Epoch %d"%epoch)
        for idx,(labels,targets) in enumerate(tqdm.tqdm(trainloader)):
            preds,_ = model(labels)
            targets = to_one_hot(targets)
            loss = (targets*torch.nn.functional.relu(.9-preds).pow(2) + .5 * (1-targets)*torch.nn.functional.relu(preds-.1).pow(2)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total = 0
        total_correct = 0
        for idx,(labels,targets) in enumerate(validloader):
            preds_onehot,_ = model(labels)
            _,preds  = torch.max(preds_onehot,dim=1)
            total += len(preds)
            total_correct += (preds == targets).sum().data.cpu().numpy()

        print(total,total_correct)
