###############################################################################
# Iffley route modelling
#
# This file generates new sentences sampled from the model
#
###############################################################################

import argparse

import torch
import json
import reader

parser = argparse.ArgumentParser(description='PyTorch Iffley Route Generating Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='100',
                    help='number of routes to generate. Note that actual number might be lower due to filtering of routes that already exist')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed, change to generate different routes')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=0.5,
                    help='temperature - higher will increase diversity, lower increase confidence. 0.5 is recommended')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = reader.Corpus(args.data)
ntokens = len(corpus.dictionary)

existing_routes = json.load(open("data/train.json"))+json.load(open("data/dev.json"))
existing_routes = [set(x[0]) for x in existing_routes]
with open(args.outf, 'w') as outf:
    for i in range(args.words):
        hidden = model.init_hidden(1)
        #initialize with <sos>
        input = torch.ones((1,1),dtype=torch.long).to(device)#torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
        with torch.no_grad():  # no tracking history
            route = []
            for i in range(25):
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

                word = corpus.dictionary.idx2word[word_idx]
                if word=="<eos>":
                    break
                route.append(word)
            #start is token that means start of climbing. Everything before start is considered a standing start, start at the beginning means sit start
            if route.count("start")!=1:
                continue
            route_no_doubles = []
            for i in range(len(route)):
                if not route[i] in route[:i]:
                    route_no_doubles.append(route[i])
            route = route_no_doubles

            if set(route) in existing_routes:
                continue
            existing_routes.append(set(route))
            if route[0]=="start":
                outf.write(' '.join(route[1:])+"\n")
            else:
                start_loc = route.index("start")
                outf.write("("+' '.join(route[:start_loc])+") "+' '.join(route[start_loc+1:])+"\n")


                #if i % args.log_interval == 0:
                #    print('| Generated {}/{} words'.format(i, args.words))
