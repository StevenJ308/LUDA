import shutil
import torch.utils.data
from .base import Trainer
from copy import deepcopy
from .helper import *
from utils import *
from dataloader.data_utils import *
from .Network import *


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_neg(self):
        if self.args.dataset == 'cifar100':
            optimizer = torch.optim.SGD(
                [{'params': self.model.encoder.layer1.parameters(), 'lr': self.args.lr_scale ** 0 * self.args.lr_neg},
                 {'params': self.model.encoder.layer2.parameters(), 'lr': self.args.lr_scale ** 1 * self.args.lr_neg},
                 {'params': self.model.encoder.layer3.parameters(), 'lr': self.args.lr_scale ** 2 * self.args.lr_neg},
                 {'params': self.model.fc.parameters(), 'lr': self.args.lr_neg}],
                momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        else:
            optimizer = torch.optim.SGD(
                [{'params': self.model.encoder.layer1.parameters(), 'lr': self.args.lr_scale ** 0 * self.args.lr_neg},
                 {'params': self.model.encoder.layer2.parameters(), 'lr': self.args.lr_scale ** 1 * self.args.lr_neg},
                 {'params': self.model.encoder.layer3.parameters(), 'lr': self.args.lr_scale ** 2 * self.args.lr_neg},
                 {'params': self.model.encoder.layer4.parameters(), 'lr': self.args.lr_scale ** 3 * self.args.lr_neg},
                 {'params': self.model.fc.parameters(), 'lr': self.args.lr_neg}],
                momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        # optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_neg, momentum=0.9, nesterov=True,
        #                             weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step,
                                                        gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.n_epochs)

        return optimizer, scheduler

    def get_optimizer_base(self):
        if self.args.dataset == 'cifar100':
            optimizer = torch.optim.SGD(
                [{'params': self.model.encoder.layer1.parameters(), 'lr': self.args.lr_scale ** 2 * self.args.lr_base},
                 {'params': self.model.encoder.layer2.parameters(), 'lr': self.args.lr_scale ** 1 * self.args.lr_base},
                 {'params': self.model.encoder.layer3.parameters(), 'lr': self.args.lr_scale ** 0 * self.args.lr_base},
                 {'params': self.model.fc.parameters(), 'lr': self.args.lr_base}],
                momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        else:
            optimizer = torch.optim.SGD(
                [{'params': self.model.encoder.layer1.parameters(), 'lr': self.args.lr_scale ** 3 * self.args.lr_base},
                 {'params': self.model.encoder.layer2.parameters(), 'lr': self.args.lr_scale ** 2 * self.args.lr_base},
                 {'params': self.model.encoder.layer3.parameters(), 'lr': self.args.lr_scale ** 1 * self.args.lr_base},
                 {'params': self.model.encoder.layer4.parameters(), 'lr': self.args.lr_scale ** 0 * self.args.lr_base},
                 {'params': self.model.fc.parameters(), 'lr': self.args.lr_base}],
                momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        # optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9,
        #                             nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        elif self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step,
                                                        gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, unsuperset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, unsuperset, trainloader, testloader = get_new_dataloader(self.args, session)

        return trainset, unsuperset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()
        result_list = [args]

        # masknum = 3
        # mask = np.zeros((args.base_class, args.num_classes))
        # for i in range(args.num_classes - args.base_class):
        #     picked_dummy = np.random.choice(args.base_class, masknum, replace=False)
        #     mask[:, i + args.base_class][picked_dummy] = 1
        # mask = torch.tensor(mask).cuda()

        for session in range(args.start_session, args.sessions):

            train_set, unsuperset, trainloader, testloader = self.get_dataloader(session)

            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label
                print('new classes for this session:\n', np.unique(train_set.targets))
                # train_set1 = deepcopy(train_set)
                # if args.use_pub:
                #     if args.select_num:
                #         train_set1 = pub_pseudo_label(unsuperset, train_set1, args)
                #     else:
                #         pass
                # else:
                #     unsuperset = pseudo_label(unsuperset)
                #     train_set1 = data_concate(train_set1, unsuperset, args)

                # trainloader1 = torch.utils.data.DataLoader(dataset=train_set1, batch_size=args.batch_size_base,
                #                                            shuffle=True, num_workers=0, pin_memory=True)
                # print('Start Warm-Up')
                # optimizer_init = torch.optim.SGD(self.model.parameters(), 0.01, momentum=0.9,
                #                             nesterov=True, weight_decay=self.args.decay)
                # for init_epoch in range(30):
                #     tl, ta = warm_train(self.model, trainloader1, optimizer_init, init_epoch, args)

                print("*************START NEG_TRAIN***************")
                optimizer, scheduler = self.get_optimizer_neg()
                for n_epoch in range(args.n_epochs):
                    tl, ta = negative_pretrain(self.model, trainloader, optimizer, scheduler,  n_epoch, args)
                    tsl, tsa = test(self.model, testloader, n_epoch, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = n_epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))
                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            n_epoch, lrc, tl, ta, tsl, tsa))
                    scheduler.step()

                optimizer1, scheduler1 = self.get_optimizer_base()
                self.model.load_state_dict(self.best_model_dict)
                print("*************START BASE_TRAIN***************")
                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    tl, ta = base_train(self.model, trainloader, optimizer1, scheduler1, epoch, args)
                    # test model with all seen class
                    tsl, tsa = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer1.state_dict(), os.path.join(args.save_path, 'optimizer_best_1.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler1.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler1.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.update_fc(trainloader, np.unique(train_set.targets), session)
                tsl, tsa = test(self.model, testloader, 0, args, session, validation=False)
                # tsl, tsa = self.test_intergrate(self.model, testloader, 0, args, session, validation=True)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def test_intergrate(self, model, testloader, epoch, args, session, validation=True):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        va5 = Averager()
        lgt = torch.tensor([])
        lbs = torch.tensor([])
        proj_matrix = torch.mm(self.dummy_classifiers,
                               F.normalize(torch.transpose(model.fc.weight[:test_class, :], 1, 0), p=2, dim=-1))

        eta = args.eta

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                emb = model.encode(data)

                proj = torch.mm(F.normalize(emb, p=2, dim=-1), torch.transpose(self.dummy_classifiers, 1, 0))
                topk, indices = torch.topk(proj, 40)
                res = (torch.zeros_like(proj))
                res_logit = res.scatter(1, indices, topk)

                logits1 = torch.mm(res_logit, proj_matrix)
                logits2 = model.forpass_fc(data)[:, :test_class]
                logits = eta * F.softmax(logits1, dim=1) + (1 - eta) * F.softmax(logits2, dim=1)

                loss = F.cross_entropy(logits, test_label)
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

        return vl, va

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        cur_time = time.localtime()
        day = cur_time.tm_mday
        hour = cur_time.tm_hour
        minute = cur_time.tm_min
        save_time = str(day) + '-' + str(hour) + '-' + str(minute)
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'
        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)

        self.args.save_path = self.args.save_path + 'Epo_%d-Bs_%d-time_%s' % (
            self.args.epochs_base, self.args.batch_size_base, save_time)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        current_file = os.path.abspath(__file__)
        shutil.copy2(current_file, self.args.save_path)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        source_file_name = 'helper.py'
        source_file = os.path.join(current_dir, source_file_name)
        shutil.copy2(source_file, self.args.save_path)
        return None
