import torch
from utils import accuracy

class AccuracyTester:
    def __init__(self):
        self._create_input()

    def _create_input(self):
        torch.manual_seed(42)
        batch_size = 32
        num_class = 10

        output = torch.randn(batch_size, num_class)
        target = torch.randint(0, num_class, (batch_size, ))

        top1_acc, top3_acc = accuracy(output, target, top_k=(1, 3))

        print(top1_acc, top3_acc)

            

if __name__ == "__main__":
    main = AccuracyTester()