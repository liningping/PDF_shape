import os
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from src.data.helpers import get_data_loaders, get_data_loaders_sample_level
from src.models import get_model
from src.utils.logger import create_logger

import torch.optim as optim
from pytorch_pretrained_bert import BertAdam

from sklearn.metrics import f1_score, accuracy_score
from src.utils.utils import *

sample_nums = []

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=16)
    parser.add_argument("--bert_model", type=str, default="./bert-base-uncased")#, choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="/path/to/data_dir/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="bow", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt","latefusion_pdf", "latefusion_shape"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/path/to/save_dir/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="food101", choices=["food101","MVSA_Single"])
    parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--df", type=bool, default=True)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--baseline", type=str)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=2)

def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay":args.weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    else:
        # param_optimizer = list(model.named_parameters())
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        param_optimizer = list(model.named_parameters())
        decay=["ConfidNet_txt.0.weight","ConfidNet_img.1.weight"]
        name = [n for n, p in param_optimizer if any(nd in n for nd in decay)]
        print(name)
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in decay)], "weight_decay": 0.0, },
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr)
    return optimizer

def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

def warmup_epoch(args, model, dataloader, optimizer):
    criterion = get_criterion(args)
    model.train()
    print("Warm up ... ")

    _loss = 0

    for (txt, segment, mask, img, tgt, idx) in tqdm(dataloader, total=len(dataloader)):
        optimizer.zero_grad()
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        tgt = tgt.cuda()
        out, txt_out, img_out = model(txt, mask, segment, img, 'pdf_train')

        loss = criterion(out, tgt)
        loss.backward()

        optimizer.step()

        _loss += loss.item()

    return _loss/len(dataloader)


def train_epoch(args, model, dataloader, optimizer):
    criterion = get_criterion(args)
    model.train()
    print("Train ... ")

    _loss = 0

    for (txt, segment, mask, img, tgt, idx) in tqdm(dataloader, total=len(dataloader)):
        optimizer.zero_grad()
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        tgt = tgt.cuda()
        out, txt_out, img_out = model(txt, mask, segment, img, 'shape_train')

        loss = criterion(out, tgt)
        loss.backward()

        optimizer.step()

        _loss += loss.item()

    sample_nums.append(len(dataloader))

    return _loss/len(dataloader)

def execute_modulation_sample_level(args, model, dataloader, epoch):
    contribution = {}
    softmax = nn.Softmax(dim=1)
    con_txt = 0.0
    con_img = 0.0

    with torch.no_grad():
        model.eval()

        for (txt, segment, mask, img, tgt, idx) in dataloader:
            txt, img = txt.cuda(), img.cuda()
            mask, segment = mask.cuda(), segment.cuda()
            tgt = tgt.cuda()
            out, txt_out, img_out = model(txt, mask, segment, img, 'shape_train')

            prediction = softmax(out)
            prediction_txt = softmax(txt_out)
            prediction_img = softmax(img_out)

            for i, item in enumerate(tgt):
                index_all = torch.argmax(prediction[i])
                index_txt = torch.argmax(prediction_txt[i])
                index_img = torch.argmax(prediction_img[i])
                value_all = 0.0
                value_txt = 0.0
                value_img = 0.0

                if index_all == tgt[i]:
                    value_all = 2.0
                if index_txt == tgt[i]:
                    value_txt = 1.0
                if index_img == tgt[i]:
                    value_img = 1.0

                contrib_txt = (value_all + value_txt - value_img) / 2.0
                contrib_img = (value_all + value_img - value_txt) / 2.0

                con_txt += contrib_txt
                con_img += contrib_img

                contribution[int(idx[i])] = (contrib_txt, contrib_img)

    con_txt = con_txt / len(dataloader)
    con_img = con_img / len(dataloader)

    train_dataloader = None
    if epoch >= args.warmup_epochs - 1:
        train_dataloader = get_data_loaders_sample_level(args, contribution)

    return con_txt, con_img, train_dataloader

def model_forward(i_epoch, model, args, criterion, optimizer, batch, mode='eval'):
    txt, segment, mask, img, tgt, idx = batch
    freeze_img = i_epoch < args.freeze_img
    freeze_txt = i_epoch < args.freeze_txt

    if args.model == "bow":
        txt = txt.cuda()
        out = model(txt)
    elif args.model == "img":
        img = img.cuda()
        out = model(img)
    elif args.model == "concatbow":
        txt, img = txt.cuda(), img.cuda()
        out = model(txt, img)
    elif args.model == "bert":
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        out = model(txt, mask, segment)
    elif args.model == "concatbert":
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out = model(txt, mask, segment, img)

    elif args.model == "latefusion_pdf":
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        tgt = tgt.cuda()
        maeloss = nn.L1Loss(reduction='mean')
        out, txt_logits, img_logits, txt_tcp_pred, img_tcp_pred = model(txt, mask, segment,img,'pdf_train')
        label = F.one_hot(tgt, num_classes=args.n_classes)  # [b,c]

        if args.task_type == "multilabel":
            txt_pred = torch.sigmoid(txt_logits)
            img_pred = torch.sigmoid(img_logits)
        else:
            txt_pred = torch.nn.functional.softmax(txt_logits, dim=1)
            img_pred = torch.nn.functional.softmax(img_logits, dim=1)
        txt_tcp, _ = torch.max(txt_pred * label, dim=1,keepdim=True)
        img_tcp, _ = torch.max(img_pred * label, dim=1,keepdim=True)

    elif args.model == "latefusion_shape":
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        tgt = tgt.cuda()
        out, txt_logits, img_logits = model(txt, mask, segment, img,'shape_train')
        label = F.one_hot(tgt, num_classes=args.n_classes)  # [b,c]

    else:
        assert args.model == "mmbt"
        for param in model.enc.img_encoder.parameters():
            param.requires_grad = not freeze_img
        for param in model.enc.encoder.parameters():
            param.requires_grad = not freeze_txt

        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out = model(txt, mask, segment, img)

    tgt = tgt.cuda()

    txt_clf_loss = nn.CrossEntropyLoss()(txt_logits, tgt)
    img_clf_loss = nn.CrossEntropyLoss()(img_logits, tgt)
    clf_loss=txt_clf_loss+img_clf_loss+nn.CrossEntropyLoss()(out,tgt)

    if mode=='train':
        # loss = torch.mean(clf_loss)+torch.mean(tcp_pred_loss)
        loss = torch.mean(clf_loss)
        return loss,out,tgt
    else:
        # loss= torch.mean(clf_loss)+torch.mean(tcp_pred_loss)
        loss= torch.mean(clf_loss)
        return loss,out,tgt


def model_eval(i_epoch, data, model, args, criterion,optimizer, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, optimizer, batch, mode='eval')
            losses.append(loss.item())

            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics

def train(args):
    con_txt = []
    con_img = []
    set_seed(5)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)
    print(args.df)

    train_loader, val_loader, test_loader = get_data_loaders(args, shuffle=True)
    train_val_loader, _, _ = get_data_loaders(args, shuffle=False)

    model = get_model(args)
    
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf
    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    logger.info("Training..")
    logger.info(model)

    for epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()

        print("Epoch: {}: ".format(epoch))
        if epoch < args.warmup_epochs:
            batch_loss = warmup_epoch(
                args, model, train_loader, optimizer
            )
        else:
            batch_loss = train_epoch(
                args, model, train_loader, optimizer
            )

        if epoch >= args.warmup_epochs - 1:
            con_t, con_i, train_loader = execute_modulation_sample_level(
                args, model, train_val_loader, epoch
            )
        else:
            con_t, con_i, _ = execute_modulation_sample_level(
                args, model, train_val_loader, epoch
            )
        con_txt.append(con_t)
        con_img.append(con_i)
        train_losses.append(batch_loss)
        model.eval()
        metrics = model_eval(epoch, val_loader, model, args, criterion, optimizer)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )
        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    for test_name, test_loader in test_loader.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion, optimizer,store_preds=True
        )
        log_metrics(f"Test - {test_name}", test_metrics, args, logger)
        

def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()