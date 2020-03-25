"""import torch
import torchvision
from parser import SGDParser
from distributed_sgd import DistributedSGD

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(optimizer.profile)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    path = '/mnt/data'
    cifar100_training = torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(cifar100_training, shuffle=True, 
    num_workers=num_workers, batch_size=batch_size, batch_size=args.batch_size)
    cifar100_testing = torchvision.datasets.CIFAR100(root=path, train=False, transform=transform_train)
    cifar100_testing_loader = DataLoader(cifar100_training, shuffle=True, 
    num_workers=num_workers, batch_size=args.test_batch_size, batch_size=args.batch_size)
    model = torchvision.models.resnet50()
    optimizer = DistributedSGD(model.parameters(), lr=args.lr, dtype=args.dtype, backend=args.backend)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, cifar100_training_loader, optimizer, epoch)
        test(args, model, device, cifar100_testing_loader)
"""