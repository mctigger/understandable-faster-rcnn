import torch


class CategoricalAccuracy(torch.nn.Module):
    def __init__(self, top_k=1):
        super(CategoricalAccuracy, self).__init__()
        self.top_k = top_k
        self._name = 'acc_metric'

    def forward(self, pred, target):
        top_k = pred.topk(self.top_k, 1)[1]
        true_k = target.view(len(target), 1).expand_as(top_k)
        correct_count = top_k.eq(true_k).float().sum().item()
        total_count = len(pred)
        accuracy = 100. * float(correct_count) / float(total_count)
        return accuracy