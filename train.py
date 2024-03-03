import os, sys
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

@hydra.main(config_path="configs/", config_name="train.yaml", version_base=None)
def main(config: DictConfig):

    # 加载数据
    train_data = hydra.utils.instantiate(config.dataset.train_dataset)
    close_test_data = hydra.utils.instantiate(config.dataset.close_test_dataset)
    open_test_data = hydra.utils.instantiate(config.dataset.open_test_dataset)
    trainloader = DataLoader(train_data, config.train_bs, shuffle=True, num_workers=os.cpu_count()//2, pin_memory=True)
    close_testloader = DataLoader(close_test_data, config.test_bs, num_workers=os.cpu_count()//2)
    open_testloader = DataLoader(open_test_data, config.test_bs, num_workers=os.cpu_count()//2)

    # 加载model
    model = hydra.utils.instantiate(config.model)
    
    # 加载损失
    loss = hydra.utils.instantiate(config.criterion)

    # 加载优化器和学习率调度器
    optimizer = hydra.utils.instantiate(config.optim, model.get_training_modules)
    scheduler = hydra.utils.instantiate(config.scheduler, optimizer)
    
    # 加载训练器
    trainer = hydra.utils.instantiate(config.trainer, config)
    
    # 开始训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for epoch in range(1, config.epochs+1):
        for batch_id, batch in enumerate(trainloader):
            trainer.train_one_batch(model, batch, loss, optimizer, device)


    # 加载评估器

    



if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True
    # ROOTDIR = os.environ.get('ROOTDIR', '.')
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['OC_CAUSE'] = '1'
    OmegaConf.register_new_resolver("eval", eval)
    # engine.default_workers = min(16, os.cpu_count())
    main()