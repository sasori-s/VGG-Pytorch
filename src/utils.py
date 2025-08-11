import torch


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

