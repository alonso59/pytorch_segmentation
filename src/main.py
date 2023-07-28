import os
import yaml
import torch
import torch.nn as nn
import logging

from models import SegmentationModels
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from utils.initialize import initialize as init

from training.dataset import data_loaders
from training.trainer import Trainer, evaluation
from training.scheduler import CyclicCosineDecayLR
from training.loss import WeightedCrossEntropyDice, CrossEntropyLoss, LogCoshDice
from monai.losses.dice import DiceFocalLoss, DiceLoss

def train(cfg):
    logger, checkpoint_path, log_path = init(cfg)
    paths = cfg['paths']
    hyper = cfg['hyperparameters']
    general = cfg['general']
    # Hyperparameters
    batch_size = hyper['batch_size']
    num_epochs = hyper['num_epochs']
    lr = hyper['lr']
    B1 = hyper['b1']
    B2 = hyper['b2']
    weight_decay = hyper['weight_decay']
    n_gpus = cfg['hyperparameters']['n_gpus']
    # Paths
    base = paths['data_base']
    train_imgdir = os.path.join(base, paths['train_imgdir'])
    train_mskdir = os.path.join(base, paths['train_mskdir'])
    val_imgdir = os.path.join(base, paths['val_imgdir'])
    val_mskdir = os.path.join(base, paths['val_mskdir'])
    pretrain = general['pretrain']
    checkpoint = general['checkpoint']
    # General settings
    n_classes = general['n_classes']
    img_sizeh = general['img_sizeh']
    img_sizew = general['img_sizew']
    channels = general['channels']
    name_model = cfg['model_name']
    device = torch.device(general['device'])
    # Getting loader
    train_loader, val_loader = data_loaders(train_imgdir=train_imgdir,
                                            train_maskdir=train_mskdir,
                                            val_imgdir=val_imgdir,
                                            val_maskdir=val_mskdir,
                                            batch_size=batch_size,
                                            num_workers=12
                                            )
    # for f in data_augmentation.transforms:
    #     op = json.dumps(f.get_transform_init_args())
    #     logger.info(f'{f.__class__.__name__}, p={f.p}, {op}')
    logger.info(f'Training items: {len(train_loader) * batch_size}')
    logger.info(f'Validation items: {len(val_loader) * batch_size}')
    logger.info(f'Factor train/val: {len(train_loader) * batch_size / (len(val_loader) * batch_size + len(train_loader) * batch_size)}')

    # Building model
    models_class = SegmentationModels(device, config_file=cfg, in_channels=channels,
                                      img_sizeh=img_sizeh, img_sizew=img_sizew,
                                      n_classes=n_classes, pretrain=pretrain)

    model, name_model = models_class.model_building(name_model=name_model)

    models_class.summary(logger=logger)

    if n_gpus > 1:
        print("#### Data parallel... ##### GPU count: ", n_gpus)
        model = nn.DataParallel(model, device_ids=[x for x in range(n_gpus)])

    if checkpoint:
        logger.info('Loading weights...')
        pretrained_path = cfg['general']['init_weights']
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if n_gpus > 1:
            for key in list(pretrained_dict.keys()):
                pretrained_dict[key.replace('layers.', 'module.layers.')] = pretrained_dict.pop(key)
        model.load_state_dict(pretrained_dict, strict=True)

    # Prepare training
    if hyper['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))
    elif hyper['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=B1)
    else:
        raise AssertionError('Optimizer not implemented')

    if hyper['loss_fn'] == 'dice_loss':
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    if hyper['loss_fn'] == 'wce_dice':
        weights = [ 0.20960683, 22.85769936, 14.34060749, 26.39703168, 12.85362307]
        weights = [1, 1, 1, 1, 1]
        loss_fn = WeightedCrossEntropyDice(device=device, lambda_=0.6, class_weights=weights) #[0.21011505, 22.14241546, 14.21335034, 23.14514484, 12.19832582]
    if hyper['loss_fn'] == 'dice_focal_loss':
        loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, lambda_dice=0.7, lambda_focal=0.3)
    if hyper['loss_fn'] == 'log_cosh_dice':
        loss_fn = LogCoshDice(device=device)
    if hyper['loss_fn'] == 'ce':
        loss_fn = CrossEntropyLoss(device=device, weights=[1 for _ in range(n_classes)])


    if hyper['scheduler']['type'] == 'step' and hyper['scheduler']['type'] is not None:
        scheduler = StepLR(
            optimizer=optimizer, step_size=cfg['hyperparameters']['scheduler']['step'],
            gamma=cfg['hyperparameters']['scheduler']['gamma'])
    elif hyper['scheduler']['type'] == 'cosine' and hyper['scheduler']['type'] is not None:
        scheduler = CyclicCosineDecayLR(optimizer,
                                        init_decay_epochs=150,
                                        min_decay_lr=0.0001,
                                        restart_interval=50,
                                        restart_lr=0.0008)
    elif hyper['scheduler']['type'] == 'exponential' and hyper['scheduler']['type'] is not None:
        scheduler = ExponentialLR(optimizer, gamma=cfg['hyperparameters']['scheduler']['gamma'], last_epoch=-1)
    # """ Trainer """
    logger.info('**********************************************************')
    logger.info('**************** Initialization sucessful ****************')
    logger.info('**********************************************************')
    logger.info('--------------------- Start training ---------------------')
    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      scheduler=scheduler, 
                      loss_fn=loss_fn, 
                      device=device, 
                      log_path=log_path, 
                      logger=logger)
    
    trainer.fit(train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                stop_value=int(num_epochs*0.2)
                )
    
    logger.info('-------------------- Finished Train ---------------------')
    logger.info('******************* Start evaluation  *******************')
    model = None
    model = torch.load(checkpoint_path + 'model.pth')
    eval = evaluation(model, val_loader, loss_fn, device)
    logger.info(eval)

if __name__ == '__main__':
    with open('configs/segmentation.yaml', "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    try:
        train(cfg)
    except Exception as exc:
        logging.error(exc)
