from dataset.dataset import IndexDataset
from mingpt.model import GPT
from torch.utils.data.dataloader import DataLoader
import time
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/500M/2.homodyne_off/output.dat"
train_split = 0.97
total_split = 1
seqlen = 20
nbits = 16
causal = False
model_type = "gpt-mini"
batch_size = 1024
save_dir = "./model/prenoise500M_homodyneoff.model"

train_dataset = IndexDataset(datadir, register_data_dir=None, split=(0,train_split), seqlen=seqlen,
                            nbits=nbits)
test_dataset = IndexDataset(datadir, register_data_dir=None, split=(train_split, total_split), seqlen=seqlen,
                            nbits=nbits)
test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    pin_memory=True,
    batch_size=batch_size,
    num_workers=8,
)

model_config = GPT.get_default_config()
model_config.model_type = model_type
model_config.vocab_size = max(train_dataset.get_vocab_size(), test_dataset.get_vocab_size())
model_config.block_size = test_dataset.get_block_size()
model = GPT(model_config)

model.load_state_dict(torch.load(save_dir))

iter_num = 0
iter_time = time.time()
data_iter = iter(test_loader)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)
losses = []
correct = 0
total = 0
y_pred = []
y_true = []
print("Begin Test")
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
    y_pred.extend(predict)
    y_true.extend(y)
    total += x.shape[0]
    losses.append(loss.item())
    iter_num += 1
    if iter_num % 100 == 0:
        print(
            f"Test iter num {iter_num}, Loss: {loss.item()}, Correct, {correct / total}")
tnow = time.time()
acc = correct / total
cf_matrix = confusion_matrix(y_true, y_pred)
per_cls_acc = cf_matrix.diagonal() / cf_matrix.sum(axis=0)
per_cls_recall = cf_matrix.diagonal() / cf_matrix.sum(axis=1)
print(
    f"Total Time:{tnow - iter_time}, "
    f"Iter num:{iter_num}, "
    f"Average Loss: {np.average(losses)}, "
    f"Accuracy: {correct / total}")

print("Precision:")
print(per_cls_acc)
print("Recall:")
print(per_cls_recall)
