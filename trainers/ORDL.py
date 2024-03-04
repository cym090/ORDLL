import torch

from trainers.base import BaseTrainer


class OrdlTrainer(BaseTrainer):
    def forward_one_batch(self, images):
        pass
    def train_one_batch(self, model, batch, criterion, optimizer, scheduler, *args, **kwargs):
        device, meters = args
        img, label = batch
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        feat = model(img)
        _, loss = criterion(feat, model.centers, model.radius1, model.radius2, label)
        # del img, feat
        # torch.cuda.empty_cache()
        loss.backward()
        optimizer.step()
        meters['loss'].update(loss.item(), img.size(0))
        for key in criterion.losses:
            meters[key].update(criterion.losses[key].item(), img.size(0))
        scheduler.step()

        