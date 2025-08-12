# All creadit goes to https://github.com/Lornatang/VGG-PyTorch/blob/main/utils.py

import torch


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


