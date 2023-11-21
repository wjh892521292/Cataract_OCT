import os
import numpy as np
import torch
from config.cfg import BaseConfig
from utils.DataAugmentation import  TrainAugmentation,TestAugmentation
from config.resnet_cfg import ResnetConfig

from utils.data6 import MyDataset

from module.trainer_f import DefaultTrainer
import torchvision.transforms as transforms

def main(args):
    runseed = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.manual_seed(runseed)
    np.random.seed(runseed)


    train_data = MyDataset(
        img_root=args.img_root,
        data_list=args.data_list,
        dataset='train',
        # dataset=args.train_csv,
        transform1=transforms.Compose([
            transforms.Resize((256, 256), interpolation=2),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        transform2=transforms.Compose([
            transforms.Resize((256, 256), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        k_fold=args.k_fold
    )

    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_data = MyDataset(
        img_root=args.img_root,
        data_list=args.data_list,
        dataset='val',
        # dataset=args.val_csv,
        transform1=transforms.Compose([
                transforms.Resize((256, 256), interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
        transform2 = transforms.Compose([
            transforms.Resize((256, 256), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        k_fold=args.k_fold
    )

    val_load = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print(len(train_load))
    print(len(val_load))
    print('load finish')


    trainer = DefaultTrainer(args)
    trainer.train(train_load, val_load)


if __name__ == '__main__':
    cfg = BaseConfig()

    fixed = None
    if cfg.parser.parse_args().exp_name == 'test':
        fixed = '--exp_name test --model_name ctt_f --max_iter 1200  --stepvalues 600 900 1100  \
              --warmup_steps 100 --save_log /data2/wangjinhong/output/wjh/save_log/Cataract_OCT/logs_catar_test/ --batch_size 8 --display_freq 1 --val_freq 10 --lr 0.1 --z_dim 128 --resnet_layers 18  --dropout 0.5 --gpu_id 0'.split()
    args = cfg.initialize(fixed)
    main(args)