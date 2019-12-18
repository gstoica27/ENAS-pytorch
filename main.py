"""Entry point."""
import os

import torch

import data
import config
import utils
import trainer
import re_trainer

from data.loader import DataLoader
from tacred_utils import scorer, constant, helper
from tacred_utils.vocab import Vocab
import numpy as np

logger = utils.get_logger()


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)

    if args.network_type == 'rnn':
        if args.dataset != 'tacred':
            dataset = data.text.Corpus(args.data_path)
        # loading tacred data
        else:
            opt = vars(args)
            opt['num_classes'] = len(constant.LABEL_TO_ID)

            # load vocab
            vocab_file = "/Volumes/External HDD/dataset/tacred/data/vocab/vocab.pkl"
            emb_file = '/Volumes/External HDD/dataset/tacred/data/vocab/embedding.npy'
            opt['data_dir'] = '/Volumes/External HDD/dataset/tacred/data/json'

            # emb_file = '/home/scratch/gis/datasets/vocab/embedding.npy'
            # vocab_file = '/home/scratch/gis/datasets/vocab/vocab.pkl'
            # opt['data_dir'] = '/home/scratch/gis/datasets/tacred/data/json'

            vocab = Vocab(vocab_file, load=True)
            opt['vocab_size'] = vocab.size
            emb_matrix = np.load(emb_file)
            assert emb_matrix.shape[0] == vocab.size
            assert emb_matrix.shape[1] == args.emb_dim

            train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False)
            score_dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=False)
            eval_dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True)
            test_batch = DataLoader(opt['data_dir'] + '/test.json', opt['batch_size'], opt, vocab, evaluation=True)

            dataset = {'train_batch': train_batch,
                       'score_dev_batch': score_dev_batch,
                       'eval_dev_batch': eval_dev_batch,
                       'test_batch': test_batch,
                       'emb_matrix': emb_matrix}
            args.num_classes = opt['num_classes']
            args.emb_matrix = emb_matrix
            args.vocab_size = opt['vocab_size']

    elif args.dataset == 'cifar':
        dataset = data.image.Image(args.data_path)
    else:
        raise NotImplementedError(f"{args.dataset} is not supported")
    if args.dataset != 'tacred':
        trnr = trainer.Trainer(args, dataset)
    else:
        trnr = re_trainer.Trainer(args, dataset)

    if args.mode == 'train':
        utils.save_args(args)
        trnr.train()
    elif args.mode == 'derive':
        assert args.load_path != "", ("`--load_path` should be given in "
                                      "`derive` mode")
        trnr.derive()
    elif args.mode == 'test':
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a "
                            "pretrained model")
        trnr.test()
    elif args.mode == 'single':
        if not args.dag_path:
            raise Exception("[!] You should specify `dag_path` to load a dag")
        utils.save_args(args)
        trnr.train(single=True)
    else:
        raise Exception(f"[!] Mode not found: {args.mode}")

if __name__ == "__main__":
    args, unparsed = config.get_args()
    print(args)
    main(args)
