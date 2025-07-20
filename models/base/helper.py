# import new Network name here and add in model_class args
# import numpy as np
import random
from copy import deepcopy
import torch.utils.data
# from torch.autograd import Function
# from torchvision.transforms import transforms

from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from models.resnet18_encoder import *
from collections import Counter


class StrongTensorTransform:
    def __init__(self):
        self.transforms = [
            self.random_horizontal_flip,
            self.random_rotation,
            self.color_jitter,
            self.add_noise,
            self.random_adjust_sharpness,
            self.random_affine
        ]

    def __call__(self, image: torch.Tensor):
        # 随机选择一些增强操作
        for transform in self.transforms:
            if random.random() < 0.5:  # 50%概率应用某个变换
                image = transform(image)
        return image

    def random_horizontal_flip(self, image: torch.Tensor):
        image = TF.hflip(image)
        return image

    def random_rotation(self, image: torch.Tensor):
        angle = random.uniform(-30, 30)  # 随机旋转角度
        image = TF.rotate(image, angle)
        return image

    def color_jitter(self, image: torch.Tensor):
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        hue = random.uniform(-0.1, 0.1)
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)
        image = TF.adjust_hue(image, hue)
        return image

    def add_noise(self, image: torch.Tensor):
        noise = torch.randn_like(image) * 0.05  # 添加噪声
        image = image + noise
        image = torch.clamp(image, 0, 1)  # 保证图像值在 [0, 1] 范围内
        return image

    def random_adjust_sharpness(self, image: torch.Tensor):
        sharpness_factor = random.uniform(0.5, 2.0)
        image = TF.adjust_sharpness(image, sharpness_factor)
        return image

    def random_affine(self, image: torch.Tensor):
        angle = random.uniform(-10, 10)
        translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
        scale = random.uniform(0.8, 1.2)
        image = TF.affine(image, angle, translate, scale, shear=0)
        return image


def pub_pseudo_label(unsuperset, train_set, args):
    unsuperload = torch.utils.data.DataLoader(dataset=unsuperset, batch_size=16, num_workers=0, shuffle=False,
                                              pin_memory=True)
    unsuperlist = []
    NET = resnet18(pretrained=True, progress=True)
    net = NET.cuda()
    net.eval()
    unsuperdata_list = []
    for i, batch1 in enumerate(unsuperload, 0):
        unsuperdata = [_.cuda() for _ in batch1]
        # for cifar100 dataset
        new_tensor = torch.nn.functional.interpolate(unsuperdata[0], size=(32, 32), mode='bilinear',
                                                     align_corners=False)
        unsuperdata_list.append(new_tensor)

        unsuperimage = net(unsuperdata[0])
        unsuperimage = unsuperimage.reshape(unsuperimage.size(0), -1)
        unsuperlist.extend(unsuperimage.detach().cpu().numpy())

    unsuperlist = np.array(unsuperlist)
    cluster_n = args.num_classes - args.base_class
    clt = KMeans(n_clusters=cluster_n, random_state=42)
    clt.fit_predict(unsuperlist)
    cluster_labels = deepcopy(clt.labels_)
    centroids = clt.cluster_centers_

    origin_label_num = Counter(cluster_labels)
    print('Origin Label Distri: ', origin_label_num)

    ori_ari = adjusted_rand_score(cluster_labels, unsuperset.targets)
    print("Origin kmeans ari: ", ori_ari)
    #
    while True:
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(cluster_n)]
        # 划分超量类和不足类
        large_clusters = [i for i, size in enumerate(cluster_sizes) if size > args.select_num]
        small_clusters = [i for i, size in enumerate(cluster_sizes) if size < args.select_num]
        if not large_clusters or not small_clusters:
            break
        for large_cluster in large_clusters:
            # 计算该簇中每个数据点到簇中心的距离
            cluster_data = unsuperlist[cluster_labels == large_cluster]
            distances = np.linalg.norm(cluster_data - centroids[large_cluster], axis=1)

            # 排序并选择远离簇中心的数据
            sorted_indices = np.argsort(distances)[::-1]
            num_to_move = int(len(cluster_data)) - args.select_num
            indices_to_move = sorted_indices[:num_to_move]

            # 选择要移动的数据点
            data_to_move = cluster_data[indices_to_move]

            # 分配数据到不足类
            for point in data_to_move:
                # 计算该点到不足类簇中心的距离
                small_centroids = centroids[small_clusters]
                distances_to_small_centroids = np.linalg.norm(point - small_centroids, axis=1)
                # 找到距离最近的不足类簇
                closest_small_cluster_index = np.argmin(distances_to_small_centroids)
                closest_small_cluster = small_clusters[closest_small_cluster_index]
                # 更新标签
                original_index = np.where((unsuperlist == point).all(axis=1))[0][0]
                cluster_labels[original_index] = closest_small_cluster

    cluster_labels = cluster_labels + args.base_class
    post_label_num = Counter(cluster_labels)
    print('PostProcess Label Distri: ', post_label_num)
    ari = adjusted_rand_score(cluster_labels, unsuperset.targets)
    print("kmeans ari: ", ari)

    if args.dataset == 'cifar100':
        unsuperdata_arr = np.array(torch.cat(unsuperdata_list, dim=0).cpu()).transpose(0, 2, 3, 1)
        train_set.data = np.vstack((train_set.data, unsuperdata_arr))
        train_set.targets = np.hstack((train_set.targets, np.array(clt.labels_)))
    else:
        train_set.data.extend(unsuperset.data)
        train_set.targets.extend(cluster_labels)
    return train_set


def negative_pretrain(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    transform = StrongTensorTransform()

    model = model.train()
    y_label = trainloader.dataset.targets
    classes = np.unique(y_label)
    if args.use_weight:
        class_counts = [sum(y_label == c) for c in classes]
        total_count = sum(class_counts)
        smoothed_weights = [np.log((total_count + 1) / (count + 0.1)) for count in class_counts]
        smoothed_weights /= np.sum(smoothed_weights)
        smoothed_weights = torch.tensor(smoothed_weights, dtype=torch.float32).cuda()
    else:
        smoothed_weights = None

    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)
        acc = count_acc(logits, train_label)

        if args.use_margin:
            phi = logits - args.neg_margin
            logits = torch.where(one_hot(train_label, logits.shape[1]).byte(), phi, logits)

        logits = args.temperature * logits
        loss = F.cross_entropy(logits, train_label.long(), weight=smoothed_weights)

        if epoch >= args.loss_iter:
            select_index = torch.where(train_label < 40)[0]
            select_data = data[select_index]
            select_label= train_label[select_index]
            # data_tmp = []
            # targets_tmp = []
            # for i in select_index:
            #     ind_cl = torch.where(i == train_label)[0]
            #     for j in ind_cl:
            #         data_tmp.append(data[j])
            #         targets_tmp.append(train_label[j])
            # data = torch.stack(data_tmp, dim=0)
            # trans_label = torch.tensor([int((x-0) * (args.num_classes-args.base_class)/args.base_class + args.base_class) for x in targets_tmp], dtype=torch.int32).cuda()

            trans_data = transform(select_data)
            trans_logits = model(trans_data)
            trans_label = select_label + args.base_class
            trans_logits = args.temperature * trans_logits
            trans_loss = F.cross_entropy(trans_logits, trans_label.long())

            # p = F.log_softmax(logits[:, :args.base_class], dim=1)
            # q = F.softmax(trans_logits[:, args.base_class:], dim=1)
            # cons_loss = F.kl_div(p, q, reduction='batchmean')
            # total_loss = loss + args.balance * trans_loss + cons_loss

            total_loss = loss + args.balance * trans_loss

        else:
            total_loss = loss

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


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()

    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        logits = model(data)
        # logits = logits[:, :args.base_class]
        acc = count_acc(logits, train_label)

        if args.use_margin:
            phi = logits - args.pos_margin
            logits = torch.where(one_hot(train_label, logits.shape[1]).byte(), phi, logits)
            # mask = train_label < args.base_class
            # logits = torch.where(mask.unsqueeze(1), phi, logits)

        logits = args.temperature * logits
        loss = F.cross_entropy(logits, train_label.long())
        total_loss = loss

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


def warm_train(model, trainloader, optimizer, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()

    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        logits = model(data)
        acc = count_acc(logits, train_label)
        logits = args.temperature * logits
        loss = F.cross_entropy(logits, train_label.long())
        total_loss = loss

        tqdm_gen.set_description(
            'Session 0, epo {}, total loss={:.4f} acc={:.4f}'.format(epoch, total_loss.item(), acc))
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
            data, label = [_.cuda() for _ in batch]
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class] = proto_list

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
            logits = args.temperature * logits
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


def compute_contrastive_loss(base_feature, new_feature, margin):
    base_center = base_feature.mean(dim=0)
    new_center = new_feature.mean(dim=0)

    distance = F.pairwise_distance(base_center.unsqueeze(0), new_center.unsqueeze(0))
    loss = torch.clamp(distance - margin, min=0).mean()
    return loss
