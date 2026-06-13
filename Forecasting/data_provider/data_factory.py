from data_provider.data_loader import Dataset_Exchange, Dataset_Weather, Dataset_ECL, Dataset_Wind
from torch.utils.data import DataLoader
from data_provider.batchsamper import FeatureBatchSampler

data_dict = {
    # 'custom': Dataset_Custom,
    'exchange': Dataset_Exchange,
    'weather': Dataset_Weather,
    'ecl': Dataset_ECL,
    'wind': Dataset_Wind
}


def data_provider(args, flag, drop_last=True, train_all=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    max_len = args.max_len
    freq = args.freq
    
    data_set = Data(
        args = args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all
    )

    sampler = FeatureBatchSampler(
        tot_len=data_set.tot_len,
        num_features=args.enc_in,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_sampler=sampler,
        num_workers=args.num_workers)
    return data_set, data_loader
