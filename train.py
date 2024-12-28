import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed
from dataset.dataset import IndexDataset
import time
from omegaconf import DictConfig, OmegaConf
import hydra
import logging

# A logger for this file
log = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s] %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="crng")
def main(cfg: DictConfig):
    # print(cfg)
    # timestamp = int(time.time())
    # logname = f"{cfg.logdir}/{cfg.name}_{timestamp}.log"
    # Remove all handlers associated with the root logger object.
    set_seed(cfg.seed)

    # datadir = "/data1/qrng/mtrng_26.dat"
    # datadir = "./data/qrng/mtrng_18.dat"
    # datadir = "./data/LCG28_24.dat"
    # datadir = "/data/hh/qrng_new/Vacuum_Fluctuation/rawdata-5-16-combine1G_150m.dat"
    # datadir = "/data/hh/qrng_new/Vacuum_Fluctuation/rawdata-5-16-combine1G_150m.dat"
    # datadir = "/data/hh/qrng_new/Vacuum_Fluctuation/rawdata-5-16-combine1G_50m.dat"
    # datadir = "./data/lehmer64_24.dat"
    # datadir = cfg["datadir"]
    # nbits = cfg["nbits"]
    # seqlen = cfg["seqlen"]
    # %%

    # print an example instance of the dataset
    train_dataset = IndexDataset(cfg.datadir, register_data_dir=None, split=(0, cfg.train_split), seqlen=cfg.seqlen,
                                 nbits=cfg.nbits)
    test_dataset = IndexDataset(cfg.datadir, register_data_dir=None, split=(cfg.train_split, cfg.total_split),
                                seqlen=cfg.seqlen,
                                nbits=cfg.nbits)

    cur_time = int(time.time())
    filename = f"{cfg.name}_{cfg.model_type}_seed_{cfg.seed}_batch_size_{cfg.batch_size}_{cur_time % 10000}"
    if not os.path.exists(cfg.logdir):
        os.makedirs(cfg.logdir)
    file_handler = logging.FileHandler(f"{cfg.logdir}/{filename}.log")
    log.addHandler(file_handler)
    # test_dataset = SortDataset('test')

    # %%

    # create a GPT instance
    from mingpt.model import GPT

    model_config = GPT.get_default_config()
    model_config.model_type = cfg.model_type
    if cfg.causal is not None:
        model_config.causal = cfg.causal
        print("Causal is set to: ", cfg.causal)
    # model_config.model_type = 'gpt-mini'
    # model_config.model_type = 'gpt-nano'
    model_config.vocab_size = max(test_dataset.get_vocab_size(), train_dataset.get_vocab_size())
    # model_config.vocab_size = train_dataset.get_vocab_size()
    # print(train_dataset.get_vocab_size())
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)

    # %%

    # create a Trainer object
    from mingpt.trainer import Trainer
    import math

    train_config = Trainer.get_default_config()
    # train_size = 10000000000
    # batch_size = 128
    # test_iter = 10000
    # train_config.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster
    train_config.max_iters = math.ceil(cfg.train_size / cfg.batch_size)
    train_config.num_workers = cfg.num_workers
    # train_config.learning_rate = 1e-3
    train_config.batch_size = cfg.batch_size
    trainer = Trainer(train_config, model, train_dataset)
    trainer.best_acc = 0

    # %%
    def test(trainer):
        if trainer.iter_num == 0 or trainer.iter_num % cfg.test_iter != 0:
            return
        test_loader = DataLoader(
            test_dataset,
            # sampler=torch.utils.data.RandomSampler(test_dataset, replacement=True, num_samples=test_dataset.size),
            shuffle=False,
            pin_memory=True,
            batch_size=cfg.batch_size // 10,
            num_workers=8,
        )

        model.train()
        iter_num = 0
        iter_time = time.time()
        data_iter = iter(test_loader)
        if train_config.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = train_config.device

        losses = []
        correct = 0
        total = 0
        log.info("Begin Test")
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            batch = [t.to(device) for t in batch]
            x, y = batch

            # forward the model
            logits, loss = model(x, y)
            # print(logits)
            predict = torch.argmax(logits, dim=-1)
            predict = predict[:, -1].detach().cpu().numpy()
            y = y[:, -1].detach().cpu().numpy()
            correct += sum(predict == y)
            total += x.shape[0]
            losses.append(loss.item())
            iter_num += 1
            if iter_num % 1000 == 0:
                log.info(
                    f"Test iter num {iter_num}, Loss: {loss.item()}, Correct, {correct / total} {correct / total * 2 ** cfg.nbits}")
        tnow = time.time()
        acc = correct / total
        if acc >= trainer.best_acc:
            trainer.best_acc = acc
            torch.save(model.state_dict(), cfg.savedir)
        log.info(
            f"Total Time:{tnow - iter_time}, "
            f"Iter num:{iter_num}, "
            f"Average Loss: {np.average(losses)}, "
            f"Accuracy: {correct / total},"
            f"Best Accuracy: {trainer.best_acc}")

    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            log.info(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; "
                f"iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}; "
                f"correct: {trainer.correct / 100 / cfg.batch_size:.5f} "
                f"{trainer.correct / 100 / cfg.batch_size * model_config.vocab_size:.3f}")
            trainer.correct = 0

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.set_callback('on_batch_end', test)
    # trainer.set_callback('on_batch_end', save)

    trainer.run()


if __name__ == "__main__":
    main()
