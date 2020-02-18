#with some code borrowed from https://github.com/pytorch/examples/tree/master/word_language_model
#best setup obtained at nhid: 100 emb: 200 lr: 1e-3 shuff: 0.1
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import random

import reader
import model

parser = argparse.ArgumentParser(description='PyTorch Language Model for Iffley boulder generation')
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=25,#leave at 25 because I hardcoded it somewhere
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--shuffle', type=float, default=0.1,
                    help='shuffle non-start tokes for data augmentation (0.0 = no shuffling)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

corpus = reader.Corpus(args.data,seqlen=args.bptt)
def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

train_batch_size=len(corpus.train)//args.bptt#hack to get sequences into one correctly ordered batch
eval_batch_size=len(corpus.valid)//args.bptt#hack to get sequences into one correctly ordered batch
train_data = batchify(corpus.train, train_batch_size)
val_data = batchify(corpus.valid, eval_batch_size)

ntokens = len(corpus.dictionary)
model = model.RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout).to(device)

optimizer = optim.Adam(model.parameters(),lr=args.lr)
criterion = nn.CrossEntropyLoss(ignore_index=0)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(train_batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in trange(1, args.epochs+1):
        epoch_start_time = time.time()
        if epoch!=1 and args.shuffle>0:
            corpus = reader.Corpus(args.data,seqlen=args.bptt, shuffle_train=args.shuffle)# data augmentation
            train_data = batchify(corpus.train, train_batch_size)
        train()
        val_loss = evaluate(val_data)
        #print('-' * 89)
        #print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                   val_loss, math.exp(val_loss)))
        #print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

print("Best val ppl: {:5.3f}".format(math.exp(best_val_loss)))

