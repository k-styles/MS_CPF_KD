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

warnings.filterwarnings("ignore")
torch.set_num_threads(1)


def train_student(args, model, data, device):
    train_dataloader, valid_dataloader, test_dataloader, _ = data
    loss_fcn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.t_epochs):
        model.train()
        loss_list = []
        for batch, batch_data in enumerate(train_dataloader):
            subgraph, feats, labels = batch_data
            feats = feats.to(device)
            labels = labels.to(device)
            model.g = subgraph
            for layer in model.gat_layers:
                layer.g = subgraph
            logits = model(feats.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()
        print(f"Epoch {epoch + 1:05d} | Loss: {loss_data:.4f}")
        if epoch % 10 == 0:
            score_list = []
            val_loss_list = []
            for batch, valid_data in enumerate(valid_dataloader):
                subgraph, feats, labels = valid_data
                feats = feats.to(device)
                labels = labels.to(device)
                score, val_loss = evaluate(feats.float(), model, subgraph, labels.float(), loss_fcn)
                score_list.append(score)
                val_loss_list.append(val_loss)
            mean_score = np.array(score_list).mean()
            print(f"F1-Score on valset  :        {mean_score:.4f} ")

            train_score_list = []
            for batch, train_data in enumerate(train_dataloader):
                subgraph, feats, labels = train_data
                feats = feats.to(device)
                labels = labels.to(device)
                train_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
            print(f"F1-Score on trainset:        {np.array(train_score_list).mean():.4f}")

    test_score_list = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, feats, labels = test_data
        feats = feats.to(device)
        labels = labels.to(device)
        test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
    print(f"F1-Score on testset:        {np.array(test_score_list).mean():.4f}")


def main(args):
    device = torch.device("cuda:0")
    data, data_info = get_data_loader(args)
    model_dict = collect_model(args, data_info)

    t1_model = model_dict['t1_model']['model']
    t2_model = model_dict['t2_model']['model']
    t3_model = model_dict['t3_model']['model']
    s_model = model_dict['s_model']['model']

    if os.path.isfile(f"{args.models_path}/t1_model.pt"):
        print("Loading t1_model")
        load_checkpoint(t1_model, "./models/t1_model.pt", device)
    else:
        print("t1_model NOT FOUND. TRAINING STARTED...")
        print("############ train teacher #############")
        train_teacher(args, t1_model, data, device)
        save_checkpoint(t1_model, "./models/t1_model.pt")
    if os.path.isfile(f"{args.models_path}/t2_model.pt"):
        print("Loading t2_model")
        load_checkpoint(t2_model, "./models/t2_model.pt", device)
    else:
        print("t2_model NOT FOUND. TRAINING STARTED...")
        print("############ train teacher #############")
        train_teacher(args, t2_model, data, device)
        save_checkpoint(t2_model, "./models/t2_model.pt")
    if os.path.isfile(f"{args.models_path}/t3_model.pt"):
        print("Loading t3_model")
        load_checkpoint(t3_model, "./models/t3_model.pt", device)
    else:
        print("t3_model NOT FOUND. TRAINING STARTED...")
        print("############ train teacher #############")
        train_teacher(args, t3_model, data, device)
        save_checkpoint(t3_model, "./models/t3_model.pt")

    print(f"number of parameter for teacher model with 1 layers: {parameters(t1_model)}")
    print(f"number of parameter for teacher model with 2 layers: {parameters(t2_model)}")
    print(f"number of parameter for teacher model with 3 layers: {parameters(t3_model)}")
    print(f"number of parameter for student model: {parameters(model_dict['s_model']['model'])}")

    if os.path.isfile(f"{args.models_path}/s_model.pt"):
        print("Loading s_model")
        load_checkpoint(s_model, "./models/s_model.pt", device)
    else:
        print("s_model NOT FOUND. TRAINING STARTED...")
        print("############ train student with teacher #############")
        train_student(args, model_dict, data, device)
        save_checkpoint(s_model, model_dict, data, device)
    
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    train_dataloader, _, test_dataloader, _ = data
    print(f"test acc of teacher with 1 layers:")
    test_model(test_dataloader, t1_model, device, loss_fcn)
    print(f"train acc of teacher with 1 layers:")
    test_model(train_dataloader, t1_model, device, loss_fcn)
    print(f"test acc of teacher with 2 layers:")
    test_model(test_dataloader, t2_model, device, loss_fcn)
    print(f"train acc of teacher with 2 layers:")
    test_model(train_dataloader, t2_model, device, loss_fcn)
    print(f"test acc of teacher with 3 layers:")
    test_model(test_dataloader, t3_model, device, loss_fcn)
    print(f"train acc of teacher with 3 layers:")
    test_model(train_dataloader, t3_model, device, loss_fcn)
    print(f"test acc of student:")
    test_model(test_dataloader, s_model, device, loss_fcn)
    print(f"train acc of student")
    test_model(train_dataloader, s_model, device, loss_fcn)



if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='GAT')
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
    parser.add_argument('--models-path', type=str, default="models/", help="Path of the directory where you have kept your models for evaluation")

    print(torch.cuda.is_available())
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)
    end = time.time()
    total_time = (end - start) / 60
    print("Total time: ", total_time, "min")
