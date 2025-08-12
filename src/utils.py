# All creadit goes to https://github.com/Lornatang/VGG-PyTorch/blob/main/utils.py

import torch
from enum import Enum


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def accuracy(output, target, top_k=(1, )):
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []

        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100 / batch_size))

        return results
    

def load_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    next_state_dict = {k : v for k, v in state_dict.items() if 
                        k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    
    model_state_dict.update(next_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def load_pretrained_state_dict(model, model_weight_path):
    checkpoints = torch.load(model_weight_path, map_location=DEVICE)
    model = load_state_dict(model, checkpoints["state_dict"])

    return model


def load_resume_state_dict(model, 
                        model_weight_path, 
                        new_model, 
                        optimizer, 
                        scheduler):
    checkpoint = torch.load(model_weight_path, map_location=DEVICE)

    start_epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_acc1"]

    model = load_state_dict(model, checkpoint["state_dict"])
    new_model = load_state_dict(new_model, checkpoint["new_model_state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, new_model, start_epoch, best_acc, optimizer, scheduler


class Summary():
    NONE = 1
    AVERAGE = 2
    SUM = 2
    COUNT = 3


class AverageMeter:
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    
    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=0):
        self.val = val
        self.sum = val * n
        self.count += n
        self.avg = self.sum / self.count


    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    

    def summary(self):
        if self.summary_type == Summary.NONE:
            fmtstr = ""
        elif self.summary_type == Summary.AVERAGE:
            fmtstr = "{name} {avg: .2f}"
        elif self.summary_type == Summary.COUNT:
            fmtstr = "{name} {count: .2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")
        
        return fmtstr.format(**self.__dict__)


