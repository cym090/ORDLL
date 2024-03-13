import os, sys
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.logger import setup_logging, record_history
import logging
from utils.misc import AverageMeter
from collections import defaultdict
from utils.test import test
import time
from utils import io
import yaml
# from utils.machine_stats import print_stats
from utils.engine import seeding

@hydra.main(config_path="configs/", config_name="train.yaml", version_base=None)
def main(config: DictConfig):
    seeding(config.seed)
    setup_logging(config.logdir + '/train.log')
    io.init_save_queue()
    # print_stats()
    # 加载数据
    logging.info(f'Load Data...')
    train_data = hydra.utils.instantiate(config.dataset.train_dataset)
    close_test_data = hydra.utils.instantiate(config.dataset.close_test_dataset)
    open_test_data = hydra.utils.instantiate(config.dataset.open_test_dataset)
    trainloader = DataLoader(train_data, config.train_bs, shuffle=True, num_workers=os.cpu_count()//2, pin_memory=True, persistent_workers=True)
    close_testloader = DataLoader(close_test_data, config.test_bs, num_workers=os.cpu_count()//4)
    open_testloader = DataLoader(open_test_data, config.test_bs, num_workers=os.cpu_count()//4)

    # 加载model
    logging.info(f'Load Model...')
    model = hydra.utils.instantiate(config.model)
    
    # 加载损失
    logging.info(f'Load Loss...')
    criterion = hydra.utils.instantiate(config.criterion, config)

    # 加载优化器和学习率调度器
    logging.info(f'Load optimizer...')
    optimizer = hydra.utils.instantiate(config.optim, model.get_training_modules())
    scheduler = hydra.utils.instantiate(config.scheduler, optimizer)
    
    # 加载训练器
    trainer = hydra.utils.instantiate(config.trainer, config)
    
    # 开始训练
    indicator = 0.0
    logging.info(yaml.dump(OmegaConf.to_object(config)))
    logging.info(f'Begin Training...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    criterion = criterion.to(device)
    start_time = time.time()
    for epoch in range(1, config.epochs+1):
        res = {'ep': epoch}
        train_meters = defaultdict(AverageMeter)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logging.info(f'Epoch [{epoch}/{config.epochs}];lr:{lr:.6}')  # 
        with tqdm(trainloader, bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
            for i, batch in enumerate(tepoch):
                trainer.train_one_batch(model, batch, criterion, optimizer, scheduler, device, train_meters)
                tepoch.set_postfix({k: v.avg for k, v in train_meters.items()})
        for key in train_meters: res['train_' + key] = train_meters[key].avg
        record_history('train', res, config.logdir)
        scheduler.step()

        # 定期评估
        eval_now = (epoch == config.epochs + 1) or (config.neval != 0 and epoch % config.neval == 0)
        if eval_now:
            results, open_k_logits, open_u_logits= test(model, criterion, close_testloader, open_testloader, config, return_logits=True)
            record_history('test', {f"ep{epoch}":results}, config.logdir)
            if config['open_test'] == True:
                open_results = results['open_out_metric']
                logging.info("Close AUROC (%): {:.3f}\t Open AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['AUROC'],open_results['AUROC'], results['close_oscr']))
                fstring1 = "Test-Epoch {}/{} Close-Open Acc:[{:.3f},{:.3f}], Close-Open AUROC:[{:.3f},{:.3f}], OSCR:{:.3f}, Close-Open TNR:[{:.3f},{:.3f}], F1[90,95,98,99],[{:.3f},{:.3f},{:.3f},{:.3f}]"
                fstring2 = "Close-Open DTACC:[{:.3f},{:.3f}], Close-Open AUIN:[{:.3f},{:.3f}], Close-Open AUOUT:[{:.3f},{:.3f}] F1:{:.3f}\n"
                fstring1 = fstring1.format(epoch,config['epochs'],results['close_acc'],results['open_acc'],results['AUROC'],open_results['AUROC'],results['close_oscr'],results['TNR'],open_results['TNR'],results['90'],results['95'],results['98'],results['99'])
                fstring2 = fstring2.format(results['DTACC'],open_results['DTACC'],results['AUIN'],open_results['AUIN'],results['AUOUT'],open_results['AUOUT'],results['f1'])
                f = fstring1 + ', ' + fstring2
                current_indicator = open_results[config.indicator]
                logging.info(f)
            else:
                logging.info("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['close_oscr']))
                logging.info("Test-Epoch {}/{}(%) Acc: {:.3f}\t AUROC: {:.3f}\t OSCR: {:.3f}\t TNR: {:.3f}\t F1(90,95,98,99):[{:.3f},{:.3f},{:.3f},{:.3f}]\t DTACC: {:.3f}\t AUIN: {:.3f}\t AUOUT: {:.3f}\n".format(epoch+1, config['max_epoch'],results['close_acc'], results['AUROC'], results['close_oscr'],
                             results['TNR'],results['f1'],results['90'],results['95'],results['98'],results['99'], results['DTACC'],results['AUIN'],results['AUOUT']))
                current_indicator = results[config.indicator]
            save_now = current_indicator > indicator
            if save_now:
                indicator = current_indicator
                model_path = f'{config.logdir}/models'
                os.makedirs(model_path, exist_ok=True)
                # trainer.save_model_state(model, model_path)
                torch.save(model.state_dict(), f"{model_path}/best.pth")
                with open(model_path+f"/{config.indicator}_{indicator}","w") as logtxt:
                    pass
                open_k_logits.tofile(f"{model_path}/open_k_logits.dat")
                open_u_logits.tofile(f"{model_path}/open_u_logits.dat")
                # save_networks(net, model_path, file_name, criterion=criterion)
    logging.info(f"End Training.Time:{time.time()-start_time:.2f}s")
    logging.info(f"Logging into {config.logdir}")
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