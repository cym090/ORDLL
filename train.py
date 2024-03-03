import os, sys
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

@hydra.main(config_path="configs/", config_name="train.yaml", version_base=None)
def main(config: DictConfig):

    # 加载数据
    train_data = hydra.utils.instantiate(config.dataset.train_data)
    close_test_data = hydra.utils.instantiate(config.dataset.close_test_data)
    open_test_data = hydra.utils.instantiate(config.dataset.open_test_data)
    trainloader = DataLoader(train_data, config.train_bs, shuffle=True, pin_memory=True)
    close_testloader = DataLoader(close_test_data, config.test_bs)
    open_testloader = DataLoader(open_test_data, config.test_bs)

    # 加载backbone
    model = hydra.utils.instantiate(config.backbone)

    # 加载损失
    pass

    # 加载优化器和学习率调度器

    # 加载训练器

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