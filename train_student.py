import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import evaluate
from utils import get_data_loader, save_checkpoint, load_checkpoint, mlp
from utils import parameters, evaluate_model, test_model, generate_label, collect_model
from loss import kd_loss, graphKL_loss, optimizing
import warnings
import datetime
import time

from data.get_cascades import load_cascades
from data.utils import load_tensor_data, initialize_label, set_random_seed, choose_path

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

# Get the current timestamp
timestamp = time.time()
# Convert the timestamp to a datetime object
dt_object = datetime.datetime.fromtimestamp(timestamp)

def train_student(args, models, data, data_info, device):
    best_score = 0
    max_score = 0
    best_loss = 1000.0

    train_dataloader, valid_dataloader, test_dataloader, fixed_train_dataloader = data

    loss_fcn = torch.nn.BCEWithLogitsLoss()

    t1_model = models['t1_model']['model']
    t2_model = models['t2_model']['model']
    t3_model = models['t3_model']['model']
    s_model = models['s_model']['model']

    step_n = 0
    alpha = 0
    lam = 7
    for epoch in range(args.s_epochs * 3):
        s_model.train()
        loss_list = []
        additional_loss_list = []
        t0 = time.time()
        #for batch, batch_data in enumerate(zip(train_dataloader, fixed_train_dataloader)):
        step_n += 1

        subgraph, feats, labels = train_dataloader.dataloader.dataset

        feats = feats.to(device)
        labels = labels.to(device)

        s_model.g = subgraph
        for layer in s_model.plp_layers:
            layer.g = subgraph

        logits, middle_feats_s, att, alpa, el, er= s_model(feats.float(), label_init=data_info['labels_init'])
        dim = len(logits[0])

        if epoch >= args.tofull:
            args.mode = 'full'

        ce_loss = loss_fcn(logits, labels.float())
        if args.mode == 'mi':
            logits_t1 = generate_label(t1_model, subgraph, feats)
            logits_t2 = generate_label(t2_model, subgraph, feats)
            logits_t3 = generate_label(t3_model, subgraph, feats)
            if epoch < args.s_epochs:
                class_loss = kd_loss(logits, logits_t1)
            elif args.s_epochs <= epoch < 2 * args.s_epochs:
                class_loss = kd_loss(logits, logits_t2)
            else:
                class_loss = kd_loss(logits, logits_t3)
            class_loss_detach = class_loss.detach()

            logits = mlp(dim, logits, device)
            logits_t1 = mlp(dim, logits_t1, device)
            logits_t2 = mlp(dim, logits_t2, device)
            logits_t3 = mlp(dim, logits_t3, device)

            if alpha == 0:
                alpha_0 = torch.mean(torch.flatten(logits.t().mm(logits_t1)))
                alpha_1 = torch.mean(torch.flatten(logits.t().mm(logits_t2)))
                alpha_2 = torch.mean(torch.flatten(logits.t().mm(logits_t3)))
                s = alpha_0 + alpha_1 + alpha_2
                alpha0 = alpha_0 / s
                alpha1 = alpha_1 / s
                alpha2 = alpha_2 / s
                sim = torch.tensor([alpha0, alpha1, alpha2])
                softmax = nn.Softmax()
                alpha_ = softmax(sim).numpy().tolist()
                alpha = [round(i * lam, 2) for i in alpha_]
            mi_loss = graphKL_loss(models, middle_feats_s[args.target_layer], subgraph, feats, class_loss_detach, epoch, args)
            if args.warmup_epoch < epoch < args.s_epochs \
                    or args.s_epochs + args.warmup_epoch <= epoch < 2 * args.s_epochs \
                    or 2 * args.s_epochs + args.warmup_epoch <= epoch < 3 * args.s_epochs:
                args.loss_weight = 0
            elif 0 <= epoch < args.warmup_epoch:
                args.loss_weight = alpha[0]
            elif args.s_epochs <= epoch < args.s_epochs + args.warmup_epoch:
                args.loss_weight = alpha[1]
            elif 2 * args.s_epochs <= epoch < 2 * args.s_epochs + args.warmup_epoch:
                args.loss_weight = alpha[2]
            additional_loss = mi_loss * args.loss_weight
        else:
            additional_loss = torch.tensor(0).to(device)

        loss = ce_loss + additional_loss

        optimizing(models, loss, ['s_model'])
        loss_list.append(loss.item())
        additional_loss_list.append(additional_loss.item() if additional_loss != 0 else 0)

        loss_data = np.array(loss_list).mean()
        additional_loss_data = np.array(additional_loss_list).mean()
        print(f"Epoch {epoch:05d} | Loss: {loss_data:.4f} | Mi: {additional_loss_data:.4f} | Time: {time.time() - t0:.4f}s")
        if epoch % 10 == 0:
            score = evaluate_model(data_info, valid_dataloader, device, s_model, loss_fcn)
            if score > best_score or loss_data < best_loss:
                best_score = score
                best_loss = loss_data
                test_score = test_model(data_info, test_dataloader, s_model, device, loss_fcn)
                if test_score > max_score:
                    max_score = test_score
    print('\033[95m' + f"f1 score on testset: {max_score:.4f}" + '\033[0m')

def load_data_idx(args):
    # load_data
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
    print('Loading cascades...')
    cas = load_cascades(cascade_dir, args.gpu, final=True)

    return G, byte_idx_train, labels_one_hot, cas


def main(args):
    device = torch.device("cuda:0")

    # load data
    data, data_info = get_data_loader(args)
    #G = byte_idx_train, labels_one_hot, cas = load_data_idx(args)
    
    model_dict = collect_model(args, data_info)

    t1_model = model_dict['t1_model']['model']
    t2_model = model_dict['t2_model']['model']
    t3_model = model_dict['t3_model']['model']
    
    with open(f'logs/{dt_object}_student.log', 'w') as file:
        if os.path.isfile("./models/t1_model.pt"):
            file.write("############ Loading teacher #############")
            file.write("\n")
            print("############ Loading teacher #############")
            load_checkpoint(t1_model, "./models/t1_model.pt", device)
        else:
            raise Exception("ERROR: Teacher NOT FOUND!")
        
        if os.path.isfile("./models/t2_model.pt"):
            file.write("############ Loading teacher #############")
            file.write("\n")
            print("############ Loading teacher #############")
            load_checkpoint(t2_model, "./models/t2_model.pt", device)
        else:
            raise Exception("ERROR: Teacher NOT FOUND!")
        
        if os.path.isfile("./models/t3_model.pt"):
            file.write("############ Loading teacher #############")
            file.write("\n")
            print("############ Loading teacher #############")
            load_checkpoint(t3_model, "./models/t3_model.pt", device)
        else:
            raise Exception("ERROR: Teacher NOT FOUND!")

        file.write(f"number of parameter for teacher model with 1 layers: {parameters(t1_model)}")
        file.write("\n")
        file.write(f"number of parameter for teacher model with 2 layers: {parameters(t2_model)}")
        file.write("\n")
        file.write(f"number of parameter for teacher model with 3 layers: {parameters(t3_model)}")
        file.write("\n")
        file.write(f"number of parameter for student model: {parameters(model_dict['s_model']['model'])}")
        file.write("\n")
        
        print(f"number of parameter for teacher model with 1 layers: {parameters(t1_model)}")
        print(f"number of parameter for teacher model with 2 layers: {parameters(t2_model)}")
        print(f"number of parameter for teacher model with 3 layers: {parameters(t3_model)}")
        print(f"number of parameter for student model: {parameters(model_dict['s_model']['model'])}")

        loss_fcn = torch.nn.BCEWithLogitsLoss()
        train_dataloader, _, test_dataloader, _ = data
        file.write(f"test acc of teacher with 1 layers:")
        file.write("\n")
        print(f"test acc of teacher with 1 layers:")
        test_model(data_info, test_dataloader, t1_model, device, loss_fcn)
        file.write(f"train acc of teacher with 1 layers:")
        file.write("\n")
        print(f"train acc of teacher with 1 layers:")
        test_model(data_info, train_dataloader, t1_model, device, loss_fcn)
        file.write(f"test acc of teacher with 2 layers:")
        file.write("\n")
        print(f"test acc of teacher with 2 layers:")
        test_model(data_info, test_dataloader, t2_model, device, loss_fcn)
        file.write(f"train acc of teacher with 2 layers:")
        file.write("\n")
        print(f"train acc of teacher with 2 layers:")
        test_model(data_info, train_dataloader, t2_model, device, loss_fcn)
        file.write(f"test acc of teacher with 3 layers:")
        file.write("\n")
        print(f"test acc of teacher with 3 layers:")
        test_model(data_info, test_dataloader, t3_model, device, loss_fcn)
        file.write(f"train acc of teacher with 3 layers:")
        file.write("\n")
        print(f"train acc of teacher with 3 layers:")
        test_model(data_info, train_dataloader, t3_model, device, loss_fcn)

        print("############ train student with teacher #############")
        train_student(args, model_dict, data, data_info, device)
        save_checkpoint(model_dict["s_model"]["model"], "./models/s_model.pt")


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--teacher", type=str, default='GAT', help="Plug in what teachers species you have trained already")
    parser.add_argument("--t-epochs", type=int, default=60, help="number of the teachers' training epochs")
    parser.add_argument("--t-num-heads", type=int, default=4, help="number of the teachers' hidden attention heads")
    parser.add_argument("--t-num-out-heads", type=int, default=6, help="number of the teachers' output attention heads")
    parser.add_argument("--t1-num-layers", type=int, default=1, help="number of teacher1's hidden layers")
    parser.add_argument("--t2-num-layers", type=int, default=2, help="number of teacher2's hidden layers")
    parser.add_argument("--t3-num-layers", type=int, default=3, help="number of teacher3's hidden layers")
    parser.add_argument("--t1-num-hidden", type=int, default=256, help="number of teacher1's hidden units")
    parser.add_argument("--t2-num-hidden", type=int, default=256, help="number of teacher2's hidden units")
    parser.add_argument("--t3-num-hidden", type=int, default=256, help="number of teacher3's hidden units")
    parser.add_argument("--s-epochs", type=int, default=500, help="number of the student's training epochs")
    parser.add_argument("--s-num-heads", type=int, default=2, help="number of the student's hidden attention heads")
    parser.add_argument("--s-num-out-heads", type=int, default=2, help="number of the student's output attention heads")
    parser.add_argument("--s-num-layers", type=int, default=4, help="number of the student's hidden layers")
    parser.add_argument("--s-num-hidden", type=int, default=68, help="number of the student's hidden units")
    parser.add_argument("--target-layer", type=int, default=1, help="the layer of student to learn")
    parser.add_argument("--mode", type=str, default='mi',
                        help="full: training student use full supervision (true label)."
                             "mi: training student use pseudo label and mutual information of middle layers.")
    parser.add_argument("--train-mode", type=str, default='together', help="training mode: together, warmup")
    parser.add_argument("--warmup-epoch", type=int, default=80, help="steps to warmup")
    parser.add_argument('--loss-weight', type=float, default=1.0, help="weight of additional loss")
    parser.add_argument('--seed', type=int, default=100, help="seed")
    parser.add_argument('--tofull', type=int, default=1500, help="change mode to full after tofull epochs")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--residual", action="store_true", default=True, help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0, help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0, help="attention dropout")
    parser.add_argument('--alpha', type=float, default=0.2, help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2, help="batch size used for training, validation and test")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0, help="weight decay")

    ### new arguments
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    parser.add_argument("--labelrate", type=int, default=20, help="Label rate")
    parser.add_argument("--student", type=str, default="PLP", help="Type of student you want to distill")
    parser.add_argument("--emb-dim", type=int, default=64, help='Embedded dim for attention')
    parser.add_argument("--feat-drop", type=float, default=0.6, help='Feature dropout for PLP')
    parser.add_argument("--plp-attn-drop", type=float, default=0.6, help='Attention dropout for PLP')
    parser.add_argument("--ptype", type=str, default="ind", help='plp type: ind(inductive); tra(transductive/onehot)')
    parser.add_argument("--mlp_layers", type=int, default=2, help='MLP layer, 0 means not add feature mlp/lr')
    parser.add_argument("--scratch", type=bool, default=False, help='Pass in True if you want to train from scratch, otherwise distill.')

    with open(f'logs/{dt_object}_student.log', 'w') as file:
        file.write(str(torch.cuda.is_available()))
        file.write("\n")
        args = parser.parse_args()
        file.write(str(args))
        file.write("\n")
    
        print(torch.cuda.is_available())
        print(args)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        main(args)
        end = time.time()
        total_time = (end - start) / 60
        file.write("Total time: {total_time} min")
        file.write("\n")
        print("Total time: ", total_time, "min")
