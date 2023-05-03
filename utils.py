import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data.ppi import LegacyPPIDataset as PPIDataset
from dgl.dataloading import GraphDataLoader

from gnns import GAT, PLP
from topo_semantic import get_loc_model, get_upsamp_model

from data.get_cascades import load_cascades
from data.utils import load_tensor_data, initialize_label, set_random_seed, choose_path, check_writable

from pathlib import Path

from extra_utils.logger import get_logger
from extra_utils.metrics import accuracy

def parameters(model):
    num_params = 0
    for params in model.parameters():
        cur = 1
        for size in params.data.shape:
            cur *= size
        num_params += cur
    return num_params

def teacher_choose_path(args):
    output_dir = Path.cwd().joinpath('outputs', args.dataset, args.teacher,
                                     'cascade_random_' + str(args.seed) + '_' + str(args.labelrate))
    check_writable(output_dir)
    cascade_dir = output_dir.joinpath('cascade')
    check_writable(cascade_dir)
    return output_dir, cascade_dir

def evaluate(data_info, feats, model, subgraph, labels, loss_fcn):
    model.eval()
    with torch.no_grad():
        model.g = subgraph
        try:
            for layer in model.plp_layers:
                layer.g = subgraph
            output = model(feats.float(), label_init=data_info['labels_init'])[0]
        except AttributeError:
            for layer in model.gat_layers:
                layer.g = subgraph
            output = model(feats.float())

        #logp = F.log_softmax(output, dim=1)
        labels = data_info['labels_one_hot']
        #print(type(output), type(labels))
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average='micro')
    model.train()

    return score, loss_data.item()


def test_model(data_info, test_dataloader, model, device, loss_fcn):
    test_score_list = []
    model.eval()
    with torch.no_grad():
        #for batch, test_data in enumerate(test_dataloader):
        subgraph, feats, labels = test_dataloader.dataloader.dataset
        feats = feats.to(device)
        labels = labels.to(device)
        test_score_list.append(evaluate(data_info, feats, model, subgraph, labels.float(), loss_fcn)[0])
        mean_score = np.array(test_score_list).mean()
        print('\033[95m' + f"F1-Score on testset:        {mean_score:.4f}" + '\033[0m')
    model.train()
    return mean_score


def generate_label(t_model, subgraph, feats, middle=False):
    t_model.eval()
    with torch.no_grad():
        t_model.g = subgraph
        for layer in t_model.gat_layers:
            layer.g = subgraph
        if not middle:
            logits_t = t_model(feats.float())
            return logits_t.detach()
        else:
            logits_t, middle_feats = t_model(feats.float(), middle)
            return logits_t.detach(), middle_feats


def evaluate_model(data_info, valid_dataloader, device, s_model, loss_fcn):
    score_list = []
    val_loss_list = []
    s_model.eval()
    with torch.no_grad():
        #for batch, valid_data in enumerate(valid_dataloader):
        subgraph, feats, labels = valid_dataloader.dataloader.dataset
        feats = feats.to(device)
        labels = labels.to(device)
        score, val_loss = evaluate(data_info, feats.float(), s_model, subgraph, labels.float(), loss_fcn)
        score_list.append(score)
        val_loss_list.append(val_loss)
    mean_score = np.array(score_list).mean()
    print(f"F1-Score on valset  :        {mean_score:.4f} ")
    s_model.train()
    return mean_score


def collate(sample):
    #print(sample)
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


def get_teacher(args, data_info):
    if (args.teacher == 'GAT'):
        heads1 = ([args.t_num_heads] * args.t1_num_layers) + [args.t_num_out_heads]
        heads2 = ([args.t_num_heads] * args.t2_num_layers) + [args.t_num_out_heads]
        heads3 = ([args.t_num_heads] * args.t3_num_layers) + [args.t_num_out_heads]
        model1 = GAT(data_info['g'], args.t1_num_layers, data_info['num_feats'], args.t1_num_hidden, data_info['n_classes'],
                     heads1, F.elu, args.in_drop, args.attn_drop, args.alpha, args.residual)
        model2 = GAT(data_info['g'], args.t2_num_layers, data_info['num_feats'], args.t2_num_hidden, data_info['n_classes'],
                     heads2, F.elu, args.in_drop, args.attn_drop, args.alpha, args.residual)
        model3 = GAT(data_info['g'], args.t3_num_layers, data_info['num_feats'], args.t3_num_hidden, data_info['n_classes'],
                     heads3, F.elu, args.in_drop, args.attn_drop, args.alpha, args.residual)
    elif (args.teacher == 'PLP'):
        # num_layers, hidden, attn_dropout, alpha, num_heads, att, layer_flag
        raise Exception("PLP Teachers not implemented yet. Only GAT Teachers.")
    return model1, model2, model3


def get_student(args, data_info):
    heads = ([args.s_num_heads] * args.s_num_layers) + [args.s_num_out_heads]
    if (args.student == 'GAT'):
        model = GAT(data_info['g'], args.s_num_layers, data_info['num_feats'], args.s_num_hidden, data_info['n_classes'],
                    heads, F.elu, args.in_drop, args.attn_drop, args.alpha, args.residual)
    elif (args.student == 'PLP'):
        print("Taking PLP as student")
        model = PLP(data_info['g'], args.s_num_layers, data_info['num_feats'], args.emb_dim, data_info['n_classes'],
                    activation=F.relu, feat_drop=args.feat_drop, attn_drop=args.plp_attn_drop, residual=False, byte_idx_train=data_info['byte_idx_train'],
                    labels_one_hot=data_info['labels_one_hot'], ptype=args.ptype, mlp_layers=args.mlp_layers)
    return model


def mlp(dim, logits, device):
    # CHANGE: Changed mlp with 1 hidden layer
    output = logits
    linear = nn.Linear(dim, dim).to(device)
    relu = nn.ReLU()
    return linear(relu(linear(output)))
    #return output


def get_feat_info(args):
    feat_info = {}
    feat_info['s_feat'] = [args.s_num_heads * args.s_num_hidden] * args.s_num_layers
    feat_info['t1_feat'] = [args.t_num_heads * args.t1_num_hidden] * args.t1_num_layers
    feat_info['t2_feat'] = [args.t_num_heads * args.t2_num_hidden] * args.t2_num_layers
    feat_info['t3_feat'] = [args.t_num_heads * args.t3_num_hidden] * args.t3_num_layers
    return feat_info


def get_data_loader(args):
    if args.dataset == 'ppi':
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4, shuffle=True)
        fixed_train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)
        
        print(dir(train_dataloader))

        n_classes = train_dataset.labels.shape[1]
        num_feats = train_dataset.features.shape[1]
        g = train_dataset.graph
        data_info = {}
        data_info['n_classes'] = n_classes
        data_info['num_feats'] = num_feats
        data_info['g'] = g
        data_info['byte_idx_train'] = None
        data_info['labels_one_hot'] = None
        data_info['labels_init'] = None
    elif args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
        if args.scratch == False:
            output_dir, cascade_dir = choose_path(args)
            
            adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
                load_tensor_data(args.student, args.dataset, args.labelrate, args.gpu)
            labels_init = initialize_label(idx_train, labels_one_hot).to(args.gpu)
            idx_no_train = torch.LongTensor(
                np.setdiff1d(np.array(range(len(labels))), idx_train.cpu())).to(args.gpu)
            byte_idx_train = torch.zeros_like(labels_one_hot, dtype=torch.bool).to(args.gpu)
            byte_idx_train[idx_train] = True
            G = dgl.graph((adj_sp.row, adj_sp.col)).to(args.gpu)
            G.ndata['feat'] = features
            G.ndata['feat'].requires_grad_()
            print('We have %d nodes.' % G.number_of_nodes())
            print('We have %d edges.' % G.number_of_edges())
            print('Not Loading cascades...')
            #cas = load_cascades(cascade_dir, args.gpu, final=True)

            #print("################## TRAINING STUDENT #####################")
            #print(f"G: {G}")
            #print(f"FEAT: {G.ndata['feat']}")
            #print(f"LABELS: {labels}")
            #print(f"byte_idx_train: {byte_idx_train}")
            #print(f"labels_one_hot: {labels_one_hot}")
            
            train_dataset = (G, G.ndata['feat'], labels_one_hot)
            valid_dataset = (G, G.ndata['feat'], labels_one_hot)
            test_dataset = (G, G.ndata['feat'], labels_one_hot)
            
            train_dataloader = GraphDataLoader(train_dataset, num_workers=0, collate_fn=collate)
            fixed_train_dataloader = GraphDataLoader(train_dataset, num_workers=0, collate_fn=collate)
            valid_dataloader = GraphDataLoader(valid_dataset, num_workers=0, collate_fn=collate)
            test_dataloader = GraphDataLoader(test_dataset, num_workers=0, collate_fn=collate)

            n_classes = int(max(labels)) - int(min(labels)) + 1
            num_feats = G.ndata['feat'].shape[1]
            
            data_info = {}
            data_info['n_classes'] = n_classes
            data_info['num_feats'] = num_feats
            data_info['g'] = G
            data_info['byte_idx_train'] = byte_idx_train 
            data_info['labels_one_hot'] = labels_one_hot 
            data_info['labels_init'] = labels_init
        else:
            output_dir, cascade_dir = teacher_choose_path(args)
            logger = get_logger(output_dir.joinpath('log'))
            #print(output_dir)
            #print(cascade_dir)
            # random seed
            torch.manual_seed(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
                load_tensor_data(args.teacher, args.dataset, args.labelrate, args.gpu)
            labels_init = initialize_label(idx_train, labels_one_hot).to(args.gpu)
            idx_no_train = torch.LongTensor(
                np.setdiff1d(np.array(range(len(labels))), idx_train.cpu())).to(args.gpu)
            byte_idx_train = torch.zeros_like(labels_one_hot, dtype=torch.bool).to(args.gpu)
            byte_idx_train[idx_train] = True
            G = dgl.graph((adj_sp.row, adj_sp.col)).to(args.gpu)
            G.ndata['feat'] = features
            print('We have %d nodes.' % G.number_of_nodes())
            print('We have %d edges.' % G.number_of_edges())
            
            #print("################## TRAINING TEACHER #####################")
            #print(f"G: {G}")
            #print(f"FEAT: {G.ndata['feat']}")
            #print(f"LABELS: {labels}")
            
            train_dataset = [G, G.ndata['feat'], labels_one_hot]
            valid_dataset = [G, G.ndata['feat'], labels_one_hot]
            test_dataset = [G, G.ndata['feat'], labels_one_hot]
            
            train_dataloader = GraphDataLoader(train_dataset, num_workers=0)
            fixed_train_dataloader = GraphDataLoader(train_dataset, num_workers=0)
            valid_dataloader = GraphDataLoader(valid_dataset, num_workers=0)
            test_dataloader = GraphDataLoader(test_dataset, num_workers=0)

            n_classes = int(max(labels)) - int(min(labels)) + 1
            num_feats = G.ndata['feat'].shape[1]
            
            data_info = {}
            data_info['n_classes'] = n_classes
            data_info['num_feats'] = num_feats
            data_info['g'] = G
            data_info['byte_idx_train'] = byte_idx_train 
            data_info['labels_one_hot'] = labels_one_hot 
            data_info['labels_init'] = labels_init
            
    return (train_dataloader, valid_dataloader, test_dataloader, fixed_train_dataloader), data_info


def save_checkpoint(model, path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), path)
    print(f"save model to {path}")


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Load model from {path}")


def collect_model(args, data_info):
    device = torch.device("cuda:0")

    feat_info = get_feat_info(args)

    t1_model, t2_model, t3_model = get_teacher(args, data_info)
    t1_model.to(device)
    t2_model.to(device)
    t3_model.to(device)

    s_model = get_student(args, data_info)
    s_model.to(device)

    local_model = get_loc_model(feat_info)
    local_model.to(device)
    local_model_s = get_loc_model(feat_info, upsampling=True)
    local_model_s.to(device)

    upsampling_model1, upsampling_model2, upsampling_model3 = get_upsamp_model(feat_info)
    upsampling_model1.to(device)
    upsampling_model2.to(device)
    upsampling_model3.to(device)

    s_model_optimizer = torch.optim.Adam(s_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t1_model_optimizer = torch.optim.Adam(t1_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t2_model_optimizer = torch.optim.Adam(t2_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t3_model_optimizer = torch.optim.Adam(t3_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    local_model_optimizer = None
    local_model_s_optimizer = None
    upsampling_model1_optimizer = torch.optim.Adam(upsampling_model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    upsampling_model2_optimizer = torch.optim.Adam(upsampling_model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    upsampling_model3_optimizer = torch.optim.Adam(upsampling_model3.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_dict = {}
    model_dict['s_model'] = {'model': s_model, 'optimizer': s_model_optimizer}
    model_dict['local_model'] = {'model': local_model, 'optimizer': local_model_optimizer}
    model_dict['local_model_s'] = {'model': local_model_s, 'optimizer': local_model_s_optimizer}
    model_dict['t1_model'] = {'model': t1_model, 'optimizer': t1_model_optimizer}
    model_dict['t2_model'] = {'model': t2_model, 'optimizer': t2_model_optimizer}
    model_dict['t3_model'] = {'model': t3_model, 'optimizer': t3_model_optimizer}
    model_dict['upsampling_model1'] = {'model': upsampling_model1, 'optimizer': upsampling_model1_optimizer}
    model_dict['upsampling_model2'] = {'model': upsampling_model2, 'optimizer': upsampling_model2_optimizer}
    model_dict['upsampling_model3'] = {'model': upsampling_model3, 'optimizer': upsampling_model3_optimizer}
    return model_dict
