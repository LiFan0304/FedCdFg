import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from FLAlgorithms.users.userbase import User
from torch.nn.utils import clip_grad_norm_
from FLAlgorithms.servers.serverbase import Server


class UserpFedCdFg(User):
    def __init__(self,
                 args, id, model, generative_model,
                 train_data, test_data,
                 available_labels, latent_layer_idx, label_info,
                 use_adam=False, temperature=2.0, alpha=0.05):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.gen_batch_size = args.gen_batch_size
        self.generative_model = generative_model
        self.latent_layer_idx = latent_layer_idx
        self.available_labels = available_labels
        self.label_info = label_info
        # 初始化稀疏注意力机制
        self.temperature = temperature

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label: 1 for label in range(self.unique_labels)}

    def Pretrain(self, user_id, cluster_models, glob_iter, personalized=False, early_stop=100, regularization=True,
                 verbose=False):
        self.clean_up_counts()
        self.model.train()
        self.generative_model.eval()
        TEACHER_LOSS, ATTENTION_Loss, LATENT_LOSS, DISS_LOSS, CONTRASTIVE_LOSS = 0, 0, 0, 0, 0
        update_c = []
        update_f = []
        update_n = []
        self.update_data = []
        for epoch in range(self.local_epochs):
            for i in range(self.K):
                self.optimizer.zero_grad()
                #### sample from real dataset (un-weighted)
                samples = self.get_next_train_batch(count_labels=True)
                X, y = samples['X'], samples['y']
                self.update_label_counts(samples['labels'], samples['counts'])
                model_result = self.model(X, logit=True)
                # print("y",y)
                user_output_logp = model_result['output']
                user_logit_logp = model_result['logit']
                # if epoch % 5 == 0:
                update_c.append(y.detach())
                update_f.append(user_logit_logp.detach())
                predictive_loss = self.loss(user_output_logp, y)

                #### sample y and generate z
                if regularization and epoch < early_stop:
                    generative_alpha = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    generative_beta = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                    ### get generator output(latent representation) of the same label
                    flag = False
                    batch_size = 32
                    diff_input = torch.randn(batch_size, batch_size)
                    perturbation = torch.zeros_like(diff_input)
                    gen_output = self.generative_model(y, flag, perturbation, latent_layer_idx=self.latent_layer_idx)[
                        'output']

                    # print("gen_output0",gen_output,"size",gen_output.size())
                    logit_given_gen = self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['logit']
                    # print("1",logit_given_gen)
                    # if epoch % 5 == 0:
                    update_f.append(logit_given_gen.detach())
                    target_p = F.softmax(logit_given_gen + 1e-10, dim=1)
                    # print("target_p",target_p)
                    user_latent_loss = generative_beta * self.ensemble_loss(user_output_logp, target_p)

                    sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)
                    sampled_y = torch.tensor(sampled_y)
                    gen_result = self.generative_model(sampled_y, flag, perturbation,
                                                       latent_layer_idx=self.latent_layer_idx)
                    # print("gen_result",gen_result,"size",len(gen_result))
                    gen_output = gen_result['output']  # latent representation when latent = True, x otherwise
                    user_output_logp = self.model(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    teacher_loss = generative_alpha * torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    # this is to further balance oversampled down-sampled synthetic data    gen_ratio * teacher_loss
                    gen_ratio = self.gen_batch_size / self.batch_size
                    loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                else:
                    #### get loss and perform optimization
                    loss = predictive_loss
                loss.backward()
                self.optimizer.step()  # self.local_model)
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        self.update_data.append(update_c)
        self.update_data.append(update_f)

    def train(self, user_id, cluster_models, cluster_weights, glob_iter, personalized=False, early_stop=100,
              regularization=True, verbose=False):
        self.clean_up_counts()
        self.model.train()
        self.generative_model.eval()
        TEACHER_LOSS, PREDICTIVE_LOSS, LATENT_LOSS, CLUSTER_LOSS, CLUSTER_LOSS1 = 0, 0, 0, 0, 0
        update_c = []
        update_f = []
        update_n = []
        self.update_data = []
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.local_epochs):
            for i in range(self.K):
                self.optimizer.zero_grad()
                #### sample from real dataset (un-weighted)
                samples = self.get_next_train_batch(count_labels=True)
                X, y = samples['X'], samples['y']
                self.update_label_counts(samples['labels'], samples['counts'])
                model_result = self.model(X, logit=True)
                # print("y",y)
                user_output_logp = model_result['output']
                user_logit_logp = model_result['logit']
                update_c.append(y.detach())
                update_f.append(user_logit_logp.detach())
                predictive_loss = self.loss(user_output_logp, y)
                # print(user_output_logp)
                cluster_total_loss = 0
                cluster_y_loss = 0
                generative_eta = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=10)
                if len(cluster_models) > 1:
                    for cluster, cluster_model in cluster_models.items():
                        cluster_model.eval()
                        cluster_result = cluster_model(X, logit=True)
                        cluster_output_logp = F.softmax(cluster_result['logit'], dim=1)
                        cluster_loss = self.ensemble_loss(user_output_logp, cluster_output_logp)

                        for cluster_id, weight in cluster_weights.items():
                            if cluster == cluster_id:
                                weights = weight
                        # weights = 1 / len(cluster_models)
                        cluster_total_loss += cluster_loss * weights
                    cluster_total_loss = cluster_total_loss * generative_eta

                #### sample y and generate z
                if regularization and epoch < early_stop:
                    generative_alpha = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    generative_beta = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)

                    ### get generator output(latent representation) of the same label
                    flag = False
                    batch_size = 32
                    diff_input = torch.randn(batch_size, batch_size)
                    perturbation = torch.zeros_like(diff_input)
                    gen_output = self.generative_model(y, flag, perturbation, latent_layer_idx=self.latent_layer_idx)[
                        'output']

                    # print("gen_output0",gen_output,"size",gen_output.size())
                    logit_given_gen = self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['logit']
                    output_given_gen = self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)[
                        'output']
                    update_f.append(logit_given_gen.detach())
                    target_p = F.softmax(logit_given_gen, dim=1)
                    # print("target_p",target_p)
                    user_latent_loss = generative_beta * self.ensemble_loss(user_output_logp, target_p)
                    # if len(cluster_models) > 1:
                    #     for cluster, cluster_model in cluster_models.items():
                    #         cluster_model.eval()
                    #         cluster_result = cluster_model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)
                    #         cluster_output_logp = F.softmax(cluster_result['logit'],dim=1)
                    #         cluster_loss = self.ensemble_loss(output_given_gen,cluster_output_logp)
                    #         # print("cluster_loss",cluster_loss)
                    #         # cluster_loss = self.loss(cluster_result['output'],y)
                    #         # cluster_y_loss += cluster_loss1
                    #         for cluster_id, weight in cluster_weights.items():
                    #             if cluster == cluster_id:
                    #                 weights = weight
                    #         # weights = 1 / len(cluster_models)
                    #         cluster_total_loss += cluster_loss * weights

                    sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)
                    sampled_y = torch.tensor(sampled_y)
                    gen_result = self.generative_model(sampled_y, flag, perturbation,
                                                       latent_layer_idx=self.latent_layer_idx)
                    gen_output = gen_result['output']  # latent representation when latent = True, x otherwise
                    user_output_logp = self.model(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    teacher_loss = generative_alpha * torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    # this is to further balance oversampled down-sampled synthetic data    gen_ratio * teacher_loss
                    gen_ratio = self.gen_batch_size / self.batch_size
                    beta = 0.1
                    loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss + beta * cluster_total_loss
                    TEACHER_LOSS += teacher_loss
                    LATENT_LOSS += user_latent_loss
                    PREDICTIVE_LOSS += predictive_loss
                    CLUSTER_LOSS += cluster_total_loss
                else:
                    #### get loss and perform optimization
                    loss = predictive_loss
                loss.backward()
                self.optimizer.step()  # self.local_model)
        # print(self.update_data)
        # local-model <=== self.model
        # print("1",self.local_model)
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        self.update_data.append(update_c)
        self.update_data.append(update_f)
        if personalized:
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        self.lr_scheduler.step(glob_iter)
        if regularization and verbose:
            TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.local_epochs * self.K)
            LATENT_LOSS = LATENT_LOSS.detach().numpy() / (self.local_epochs * self.K)
            # CLUSTER_LOSS = CLUSTER_LOSS.detach().numpy() / (self.local_epochs * self.K)
            # CLUSTER_LOSS1 = CLUSTER_LOSS1.detach().numpy() / (self.local_epochs * self.K)
            PREDICTIVE_LOSS = PREDICTIVE_LOSS.detach().numpy() / (self.local_epochs * self.K)
            info = '\nUser TEACHER Loss={:.4f}'.format(TEACHER_LOSS)
            info += ', Latent Loss={:.4f}'.format(LATENT_LOSS)
            info += ', Perdictive LOSS={:.4f}'.format(PREDICTIVE_LOSS)
            # info += ', Cluster LOSS={:.4f}'.format(CLUSTER_LOSS)
            print(info)

    def get_feature(self):
        self.model.val()
        samples = self.get_next_train_batch(count_labels=True)
        X, y = samples['X'], samples['y']
        model_result = self.model(X, logit=True)
        user_logit_logp = model_result['logit']
        return user_logit_logp

    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        # weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts])  # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights)  # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights


