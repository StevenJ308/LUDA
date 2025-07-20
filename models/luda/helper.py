# import new Network name here and add in model_class args
import torch.utils.data
from utils import *
from tqdm import tqdm
import torch.nn.functional as F


def negative_pretrain(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    extra_class = args.num_classes - args.base_class
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        logits = model(data)

        acc = count_acc(logits, train_label)

        select_index = torch.where(train_label < extra_class)[0][:int(0.4 * len(train_label))]
        select_label = train_label[select_index] + args.base_class

        if args.use_margin:
            phi_1 = logits - args.pos_margin
            logits_1 = torch.where(one_hot(train_label, logits.shape[1]).byte(), phi_1, logits)

            phi_2 = logits - args.neg_margin
            logits_2 = torch.where(one_hot(train_label, logits.shape[1]).byte(), phi_2, logits)

        logits_1 *= args.temperature
        loss = F.cross_entropy(logits_1, train_label.long())

        if epoch >= args.loss_iter:

            trans_loss = F.cross_entropy(logits_2[select_index], select_label.long())
            total_loss = loss + args.balance * trans_loss

        else:
            total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f}, acc={:.4f}'
                .format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        logits = model(data)
        if args.use_margin:
            phi = logits - args.pos_margin
            logits_ = torch.where(one_hot(train_label, logits.shape[1]).byte(), phi, logits)
        else:
            logits_ = logits

        logits_ *= args.temperature
        loss = F.cross_entropy(logits_, train_label.long())
        acc = count_acc(logits_, train_label)
        extra_class = args.num_classes - args.base_class
        select_index = torch.where(train_label < extra_class)[0][:int(0.4 * len(train_label))]
        select_label = train_label[select_index]

        trans_logits = logits[select_index]

        trans_label = select_label + args.base_class
        trans_loss = F.cross_entropy(trans_logits, trans_label.long())
        total_loss = loss + args.balance * trans_loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=0, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, train_label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(train_label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model


def test(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label.long())
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va, va5))

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs, save_model_dir)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            print('Seen Acc:', seenac, 'Unseen ACC:', unseenac)
    return vl, va


def one_hot(y, num_class):
    if type(y) != torch.int64:
        y = y.type(torch.int64)
    return torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)
