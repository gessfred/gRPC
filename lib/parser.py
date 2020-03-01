from argparse import ArgumentParser

"""
Parser that parses learning rate, gamma, ...
"""
class SGDParser(ArgumentParser):
    def __init__(self, description=''):
        super().__init__(description=description)
        self.add_argument('--dtype', default='32bit', help='level of compression. either \{1bit, 2bit, 4bit, 32bit\}')
        self.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        self.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        self.add_argument('--epochs', type=int, default=14, metavar='N',
                            help='number of epochs to train (default: 14)')
        self.add_argument('--lr', type=float, default=1.0, metavar='LR',
                            help='learning rate (default: 1.0)')
        self.add_argument('--gamma', type=float, default=0.7, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
        self.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        self.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        self.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        self.add_argument('--backend', default='gloo', help='backend for distributed communication')
        self.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')