from FLAlgorithms.users.userpFedCdFg import UserpFedCdFg
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.trainmodel.generator import Generator
from utils.model_config import GENERATORCONFIGS
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model
import torch
import math
from numpy import transpose, zeros
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import euclidean
from torchvision.utils import save_image
import os
from collections import defaultdict
import copy
from scipy.cluster.hierarchy import linkage, fcluster
from torch.optim.lr_scheduler import _LRScheduler
import time

MIN_SAMPLES_PER_LABEL = 1


class MemoryBank:
    def __init__(self, max_size=1000, min_loss_threshold=0.5):
        self.max_size = max_size
        self.min_loss_threshold = min_loss_threshold
        self.samples = []
        self.losses = []
        self.timestamps = []

    def update(self, new_samples, new_losses):
        """
        更新记忆缓冲池
        """

        new_losses = new_losses.detach()
        if new_losses.dim() == 0:
            new_losses = new_losses.unsqueeze(0)
        if isinstance(new_samples, torch.Tensor):
            new_samples = [new_samples]

        if len(new_samples) != len(new_losses):
            print(f"Warning: samples length ({len(new_samples)}) != losses length ({len(new_losses)})")
            return

        for sample, loss in zip(new_samples, new_losses):

            if isinstance(loss, torch.Tensor):
                loss_value = loss.mean().item()
            else:
                loss_value = loss

            if loss > self.min_loss_threshold:
                if len(self.samples) >= self.max_size:
                    self.samples.pop(0)
                    self.losses.pop(0)
                    self.timestamps.pop(0)
                sample = sample.detach().clone()
                if sample.dim() == 1:
                    sample = sample.unsqueeze(0)

                self.samples.append(sample)
                self.losses.append(loss.item())
                self.timestamps.append(time.time())

    def sample(self, batch_size=32):
        """
        从缓冲池中采样
        """
        if not self.samples:
            return None
        losses = [float(loss) for loss in self.losses]

        batch_size = min(batch_size, len(self.samples))
        try:
            weights = torch.tensor(losses, requires_grad=False) / sum(losses)
            indices = torch.multinomial(weights, batch_size)

            sampled_samples = [self.samples[i] for i in indices]

            sample_shapes = [s.shape for s in sampled_samples]
            if len(set(sample_shapes)) > 1:
                print(f"Warning: Different sample shapes found: {sample_shapes}")

                max_dims = max(sample_shapes)
                for i, sample in enumerate(sampled_samples):
                    if sample.shape != max_dims:

                        while len(sample.shape) < len(max_dims):
                            sample = sample.unsqueeze(0)
                        sampled_samples[i] = sample

            return sampled_samples

        except Exception as e:
            print(f"Error in memory bank sampling: {e}")
            return None

    def get_hardest_samples(self, n_samples=10):
        """
        获取最难处理的样本
        """
        if not self.samples:
            return None
        sorted_indices = np.argsort(self.losses)[-n_samples:]
        return [self.samples[i] for i in sorted_indices]

    def clear_old_samples(self, max_age=3600):
        """
        清理过旧的样本
        """
        current_time = time.time()
        indices_to_keep = [i for i, t in enumerate(self.timestamps)
                           if current_time - t < max_age]

        self.samples = [self.samples[i] for i in indices_to_keep]
        self.losses = [self.losses[i] for i in indices_to_keep]
        self.timestamps = [self.timestamps[i] for i in indices_to_keep]


class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_max = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.T_max:
            lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
                 (1 + math.cos(math.pi * epoch / self.T_max))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


class FedCdFg(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)
        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = data[0]
        total_users = len(clients)
        self.memory_bank = MemoryBank(
            max_size=args.memory_size,  # 从参数中获取
            min_loss_threshold=args.min_loss_threshold
        )
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()
        self.use_adam = 'adam' in self.algorithm.lower()
        self.early_stop = 20  # stop using generated samples after 20 local epochs
        self.student_model = copy.deepcopy(self.model)
        self.generative_model = create_generative_model(args.dataset, args.algorithm, self.model_name, args.embedding)
        if not args.train:
            print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generative_model.latent_layer_idx
        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta,
                                                                 self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn()
        self.train_data_loader, self.train_iter, self.available_labels = aggregate_user_data(data, args.dataset,
                                                                                             self.ensemble_batch_size)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=3e-4, betas=(0.9, 0.999),
            eps=1e-07, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        # self.generative_lr_scheduler = CosineAnnealingLR(optimizer=self.generative_optimizer, T_max=self.num_glob_iters)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=3e-4, betas=(0.9, 0.999),
            eps=1e-07, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)
        # self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=self.num_glob_iters)
        self.alpha = 0.05
        # self.init_optimizers_with_cosine_annealing()
        #### creating users ####
        self.users = []
        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data(i, data, dataset=args.dataset, count_labels=True)
            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)
            id, train, test = read_user_data(i, data, dataset=args.dataset)
            user = UserpFedCdFg(
                args, id, model, self.generative_model,
                train_data, test_data,
                self.available_labels, self.latent_layer_idx, label_info,
                use_adam=self.use_adam)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedAvg server.")

    def compute_sample_difficulty(self, gen_output, global_model, local_models, y, target):
        """
        计算样本的难度
        """
        difficulties = []
        gen_output = gen_output.detach().clone().requires_grad_(True)

        # 1. 计算全局模型的预测熵
        global_output = global_model(gen_output, start_layer_idx=-1, logit=True)['logit']
        global_probs = F.softmax(global_output, dim=1)
        # global_probs = torch.clamp(global_probs, min=1e-10, max=1-1e-10)
        global_entropy = -torch.sum(global_probs * torch.log(global_probs + 1e-10), dim=1)
        global_entropy = global_entropy.mean()

        # 2. 计算本地模型预测的一致性
        local_predictions = []
        for user_idx, user in enumerate(self.selected_users):
            user.model.eval()
            local_output = user.model(gen_output, start_layer_idx=-1, logit=True)['logit']
            local_probs = F.softmax(local_output, dim=1)
            local_predictions.append(local_probs)
        prediction_variance = torch.var(torch.stack(local_predictions), dim=0).mean(dim=1)
        prediction_variance = prediction_variance.mean()

        # 3. 计算梯度范数
        grad_norms = []
        for user_idx, user in enumerate(self.selected_users):
            user.model.eval()
            weight = self.label_weights[y][:, user_idx].reshape(-1, 1)
            output = user.model(gen_output, start_layer_idx=-1, logit=True)['output']
            loss = torch.mean(self.generative_model.crossentropy_loss(output, target))
            # loss.requires_grad = True
            grad = torch.autograd.grad(loss, gen_output, create_graph=True)[0]
            # grad = torch.clamp(grad, -5, 5)
            grad_norm = torch.norm(grad, dim=1)
            grad_norms.append(grad_norm)

        grad_norm = torch.mean(torch.stack(grad_norms), dim=0)
        grad_norm = torch.clamp(grad_norm, min=0, max=10)
        grad_norm = grad_norm.mean()

        # 综合计算难度
        difficulty = (global_entropy + prediction_variance + grad_norm) / 3
        
        if difficulty.dim() == 0:
            difficulty = difficulty.unsqueeze(0)
        return difficulty

    def update_memory_bank(self, gen_output, global_model, local_models, y, target):
        """
        更新记忆缓冲池
        """
        try:
            difficulties = self.compute_sample_difficulty(gen_output, global_model, local_models, y, target)

            if difficulties.dim() == 0:
                difficulties = difficulties.unsqueeze(0)
            if isinstance(gen_output, torch.Tensor):
                gen_output = [gen_output]

            # 更新记忆缓冲池
            self.memory_bank.update(gen_output, difficulties)

            # 定期清理旧样本
            if self.memory_bank.timestamps and time.time() - self.memory_bank.timestamps[0] > 3600:
                self.memory_bank.clear_old_samples()

        except Exception as e:
            print(f"Error updating memory bank: {e}")

    def compute_similarity_matrix(self):
        update_vectors = []
        for client_id, updates in self.client_updates.items():
            flat_update = []
            for name, tensor in updates['params_diff'].items():
                flat_update.append(tensor.cpu().numpy().flatten())
            flat_update = np.concatenate(flat_update)
            update_vectors.append(flat_update)
        update_matrix = np.array(update_vectors)
        similarity = cosine_similarity(update_matrix)
        return similarity

    def remove_arrays_from_update_vectors(update_vectors):
        for i, item in enumerate(update_vectors):
            if isinstance(item, np.ndarray):  
                update_vectors[i] = item.tolist()  
            elif isinstance(item, list):  
                remove_arrays_from_update_vectors(item)  
        return update_vectors

    def train(self, args):
        best_accuracy = 0
        best_model_state = None
        # 创建学习率跟踪
        lr_history = {'gen': [], 'model': []}
        #### pretraining
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ", glob_iter, " -------------\n\n")
            self.selected_users, self.user_idxs = self.select_users(glob_iter, self.num_users, return_idx=True)
            # print(self.selected_users)

            if not self.local:
                self.send_parameters(mode=self.mode)  # broadcast averaged prediction model
            self.evaluate()
            chosen_verbose_user = np.random.randint(0, len(self.users))
            self.timestamp = time.time()  # log user-training start time
            feature_u = []
            feature = []
            for user_id, user in zip(self.user_idxs, self.selected_users):  # allow selected users to train
                verbose = user_id == chosen_verbose_user
                # print("user_id,user ",user_id,user)
                # perform regularization using generated samples after the first communication round
                user.Pretrain(
                    user_id,
                    self.cluster_models,
                    glob_iter,
                    personalized=self.personalized,
                    early_stop=self.early_stop,
                    verbose=verbose and glob_iter > 0,
                    regularization=glob_iter > 0)
            update_vectors = []
            update_vectors1 = []
            update_data = []
            for user_id, user in zip(self.user_idxs, self.selected_users):
                flat_update = []
                each_update = []
                each_update1 = []
                for tensor in user.model.parameters():
                    # print(user.local_model)
                    tensor1 = tensor.detach()
                    flat_update.append(tensor1.detach().numpy().flatten())

                for data in user.update_data:
                    data_tensor = torch.cat(data, dim=0)
                    each_update.append(data_tensor.detach().numpy().flatten())
                    # for tensor in data:
                    #     each_update.append(tensor.detach().numpy().flatten())

                flat_update = np.concatenate(flat_update)
                each_update = np.concatenate(each_update)
                # print(each_update)
                flat_update = np.array(flat_update)
                each_update = np.array(each_update)
                update_vectors.append(flat_update)
                update_data.append(each_update)
                update_vectors1.append(user_id)

            
            def cosine_similarity_cluster(v1, v2):
                v1 = torch.tensor(v1)
                v2 = torch.tensor(v2)
                return cosine(v1, v2)

            def mahatun_similarity_cluster(v1, v2):
                v1 = torch.tensor(v1)
                v2 = torch.tensor(v2)
                return cityblock(v1, v2)

            def euclidean_similarity_cluster(v1, v2):
                v1 = torch.tensor(v1)
                v2 = torch.tensor(v2)
                return euclidean(v1, v2)

            distance = zeros((len(update_vectors), len(update_vectors)))
            for i in range(len(update_vectors)):
                for j in range(len(update_vectors)):
                    
                    distance[i][j] = cosine_similarity_cluster(update_vectors[i], update_vectors[j])
            distance1 = zeros((len(update_data), len(update_data)))
            for i in range(len(update_data)):
                # print("i",i,len(update_data[i]))
                for j in range(len(update_data)):
                    
                    distance[i][j] = cosine_similarity_cluster(update_vectors[i], update_vectors[j])
            # 使用层次聚类
            distance_result = self.alpha * distance + (1 - self.alpha) * distance1
            linked = linkage(distance_result, 'average')
            # # 根据阈值分割成簇
            # clusters = fcluster(linked, 3, criterion='maxclust')
            clusters = fcluster(linked, 7.0, criterion='distance')
            # print(clusters)
            cluster_groups = defaultdict(list)
            for sample, cluster in zip(update_vectors1, clusters):
                cluster_groups[cluster].append(sample)
            if len(cluster_groups) != 1 or len(cluster_groups) != 0:
                # for cluster, group in cluster_groups.items():
                # print(f"Cluster {cluster}: {group}")
                # print(update_vectors)
                # update_vectors1 = sorted(update_vectors, key=None, reverse=False)
                # print(update_vectors1)
                self.cluster_users = []
                for cluster_id, client_ids in cluster_groups.items():
                    cluster1 = []
                    cluster1.append(cluster_id)
                    for client_id in client_ids:
                        for user_id, user in zip(self.user_idxs, self.selected_users):
                            if client_id == user_id:
                                cluster1.append(user)
                    self.cluster_users.append(cluster1)
                    self.cluster_models[cluster_id] = copy.deepcopy(self.model)
                # # print("server_users",self.cluster_users)
                # # print("server",self.cluster_models)
                # for key, values in self.cluster_models.items():
                #     print(key,values)
                # for i in range(len(self.cluster_users)):
                #     print("i",self.cluster_users[i])
                #     for j in range(len(self.cluster_users[i])):
                #         if j!=0:
                #             print("j",self.cluster_users[i][j])
                #         else:
                #             id1 = self.cluster_users[i][j]
                #             print("id1",id1)
                # # self.cluster_users.append(cluster
            if len(cluster_groups) > 1:
                self.aggregate_parameters1()
                self.send_parameters1(mode=self.mode)
            else:
                self.aggregate_parameters()
                self.send_parameters(mode=self.mode)
            print("len:", len(self.cluster_models))
            for user_id, user in zip(self.user_idxs, self.selected_users):  # allow selected users to train
                verbose = user_id == chosen_verbose_user
                total_train = 0
                for user in self.selected_users:
                    total_train += user.train_samples
                for user_id, user in zip(self.user_idxs, self.selected_users):  # allow selected users to train
                    verbose = user_id == chosen_verbose_user
                    for cluster, cluster_model in self.cluster_models.items():
                        cluster_train = 0
                        for i in range(len(self.cluster_users)):
                            cluster_id = self.cluster_users[i][0]
                            if cluster == cluster_id:
                                for j in range(len(self.cluster_users[i])):
                                    if j != 0:
                                        cluster_train += self.cluster_users[i][j].train_samples
                        cluster_weight = cluster_train / total_train
                        self.cluster_weights[cluster] = cluster_weight
                user.train(
                    user_id,
                    self.cluster_models,
                    self.cluster_weights,
                    glob_iter,
                    personalized=self.personalized,
                    early_stop=self.early_stop,
                    verbose=verbose and glob_iter > 0,
                    regularization=glob_iter > 0)
            curr_timestamp = time.time()  # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)

            if self.personalized:
                self.evaluate_personalized_model()

            self.timestamp = time.time()  # log server-agg start time
            self.aggregate_parameters()
            self.train_generator(
                self.batch_size,
                glob_iter,
                epoches=self.ensemble_epochs // self.n_teacher_iters,
                latent_layer_idx=self.latent_layer_idx,
                verbose=True
            )
            # self.generative_scheduler.step()
            # self.model_scheduler.step()
            curr_timestamp = time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
            if glob_iter > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
                self.visualize_images(self.generative_model, glob_iter, repeats=10)
                
        self.save_results(args)
        self.save_model()

    def train_generator(self, batch_size, glob_iter, epoches=1, latent_layer_idx=-1, verbose=False):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return anything.
        """
        # self.generative_regularizer.train()
        self.label_weights, self.qualified_labels = self.get_label_weights()
        TEACHER_LOSS, MODEL_LOSS, DIVERSITY_LOSS1 = 0, 0, 0

        def update_generator_(y_input, y, init_flag, global_model, DIVERSITY_LOSS1):
            self.generative_model.train()
            global_model.eval()
            self.generative_optimizer.zero_grad()
            ## feed to generator
            flag = False
            if init_flag == 0:
                flag = False
            gen_result = self.generative_model(y_input, flag, self.perturbation, latent_layer_idx=latent_layer_idx,
                                               verbose=True)
            gen_output, eps = gen_result['output'], gen_result['eps']
            global_logit = global_model(gen_output, start_layer_idx=latent_layer_idx, logit=True)['logit']
            global_output = F.softmax(global_logit, dim=1)
            diversity_loss = self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs
            user_loss = 0
            user_logit = 0
            total_train = 0
            model_total_loss = 0
            for user in self.selected_users:
                total_train += user.train_samples
            for user_idx, user in enumerate(self.selected_users):
                user.model.eval()
                weight = self.label_weights[y][:, user_idx].reshape(-1, 1)
                expand_weight = np.tile(weight, (1, self.unique_labels))
                user_result_given_gen = user.model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                user_output_logp = F.softmax(user_result_given_gen['logit'], dim=1)
                user_output_logp_ = F.log_softmax(user_result_given_gen['logit'], dim=1)
                user_loss_ = torch.mean( \
                    self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                    torch.tensor(weight, dtype=torch.float32))
                user_loss += user_loss_
                # 计算熵
            entropy = -torch.sum(global_output * torch.log(global_output + 1e-10), dim=1)
            # 最大化熵（通过最小化负熵）
            g_loss = -torch.mean(entropy)
            beta = 10
            if self.ensemble_beta > 0:
                loss = self.ensemble_alpha * user_loss + self.ensemble_eta * diversity_loss + g_loss
                # loss= tea_gamma * user_loss + div_beta * diversity_loss - adv_alpha * model_total_loss
            else:
                loss = self.ensemble_alpha * user_loss + self.ensemble_eta * diversity_loss
            if torch.isnan(loss):
                print("Warning: NaN detected in generator loss!")
                return None
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.generative_optimizer.step()
            DIVERSITY_LOSS1 += diversity_loss
            return gen_output.detach(), DIVERSITY_LOSS1

        def update_global_(y_input, y, gen_output, n_iter, global_model, TEACHER_LOSS, MODEL_LOSS):

            global_model.train()
            self.generative_model.eval()
            self.optimizer.zero_grad()
            user_loss = 0
            user_logit = 0
            model_total_loss = 0
            grad_total_diff = torch.zeros_like(gen_output)
            global_output = global_model(gen_output, start_layer_idx=latent_layer_idx, logit=True)['output']
            # global_output_logp_ = F.log_softmax(global_output['logit'],dim=1)
            # print("output",global_output_logp_)
            global_loss = torch.mean(self.generative_model.crossentropy_loss(global_output, y_input))
            # print("global_loss",global_loss)
            total_train = 0
            for user in self.selected_users:
                total_train += user.train_samples
            for user_idx, user in enumerate(self.selected_users):
                user.model.eval()
                user_result_given_gen = user.model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                user_output_logp = F.softmax(user_result_given_gen['logit'], dim=1)
                model_loss = self.ensemble_loss(global_output, user_output_logp)
                weights = user.train_samples / total_train
                model_total_loss += model_loss * weights
            if self.ensemble_beta > 0:
                # loss = tea_gamma * global_loss  + adv_alpha * model_total_loss
                loss = global_loss + model_total_loss
            else:
                loss = global_loss + model_total_loss
            if torch.isnan(loss):
                print("Warning: NaN detected in generator loss!")
                return None
            loss.backward()
            self.optimizer.step()
            # print("global_loss",global_loss)
            TEACHER_LOSS += global_loss  # (torch.mean(TEACHER_LOSS.double())).item()
            MODEL_LOSS += model_total_loss  # (torch.mean(student_loss.double())).item()
            # DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#(torch.mean(diversity_loss.double())).item()
            # DIVERSITY_LOSS = 0
            return TEACHER_LOSS, MODEL_LOSS

        self.perturbation = torch.zeros_like(torch.randn(batch_size, batch_size))
        for i in range(epoches):
            for j in range(self.n_teacher_iters):
                y = np.random.choice(self.qualified_labels, batch_size)
                y_input = torch.LongTensor(y)
                gen_output, DIVERSITY_LOSS1 = update_generator_(
                    y_input, y, j, self.model, DIVERSITY_LOSS1)
                if torch.isnan(gen_output).any():
                    print("Warning: NaN detected in gen_output")
                    continue
                
                # 2. 从记忆缓冲池中采样
                memory_ratio = 0.5  # 记忆样本的比例
                memory_batch_size = int(batch_size * memory_ratio)
                memory_samples = self.memory_bank.sample(memory_batch_size)

                
                # 3. 合并新生成的样本和记忆样本
                if memory_samples is not None and len(memory_samples) > 0:
                    try:
                        
                        memory_samples = torch.stack(memory_samples)

                        if memory_samples.dim() == 3 and gen_output.dim() == 2:
                            memory_samples = memory_samples.squeeze(0)
                        elif memory_samples.dim() == 2 and gen_output.dim() == 3:
                            # 如果gen_output是3维的，需要压缩一个维度
                            gen_output = gen_output.squeeze(1)

                        # 创建新的叶子节点张量
                        gen_output = gen_output.detach().clone()
                        memory_samples = memory_samples.detach().clone()
                        
                        total_samples = batch_size
                        new_sample_size = int(total_samples * (1 - memory_ratio))
                        memory_sample_size = total_samples - new_sample_size

                        new_indices = torch.randperm(gen_output.size(0))[:new_sample_size]
                        selected_gen_output = gen_output[new_indices]
                        memory_indices = torch.randperm(memory_samples.size(0))[:memory_sample_size]
                        selected_memory_samples = memory_samples[memory_indices]
                        

                        combined_samples = torch.cat([gen_output, memory_samples], dim=0)
                        # combined_samples = combined_samples + gen_output
                        # 确保最终批次大小正确
                        assert combined_samples.size(0) == batch_size, \
                            f"Combined samples size {combined_samples.size(0)} != batch_size {batch_size}"

                    except Exception as e:
                        if isinstance(memory_samples, list):
                            print(f"Memory samples length: {len(memory_samples)}")
                            if len(memory_samples) > 0:
                                print(f"First memory sample shape: {memory_samples[0].shape}")
                        combined_samples = gen_output.detach().clone()
                else:
                    combined_samples = gen_output.detach().clone()

                # 4. 更新记忆缓冲池
                self.update_memory_bank(combined_samples, self.model, self.selected_users, y, y_input)

                TEACHER_LOSS, MODEL_LOSS = update_global_(
                    y_input, y, combined_samples, self.n_teacher_iters, self.model, TEACHER_LOSS, MODEL_LOSS)

        TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        MODEL_LOSS = MODEL_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS1 = DIVERSITY_LOSS1.detach().numpy() / (self.n_teacher_iters * epoches)
        info = "Generator: Teacher Loss= {:.4f}, MODEL_LOSS = {:.4f}, diversity loss = {:.4f} ". \
            format(TEACHER_LOSS, MODEL_LOSS, DIVERSITY_LOSS1)
        if verbose:
            print(info)
        if glob_iter % 10 == 0:
            # self.generative_lr_scheduler.step(glob_iter)
            # self.lr_scheduler.step(glob_iter)
            self.generative_lr_scheduler.step()
            self.lr_scheduler.step()

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def visualize_images(self, generator, glob_iter, repeats=1):
        """
        Generate and visualize data for a generator.
        """
        os.system("mkdir -p images")
        path = f'images/{self.algorithm}-{self.dataset}-iter{glob_iter}.png'
        y = self.available_labels
        y = np.repeat(y, repeats=repeats, axis=0)
        y_input = torch.tensor(y)
        generator.eval()
        images = generator(y_input, latent=False)['output']  # 0,1,..,K, 0,1,...,K
        images = images.view(repeats, -1, *images.shape[1:])
        images = images.view(-1, *images.shape[2:])
        save_image(images.detach(), path, nrow=repeats, normalize=True)
        print("Image saved to {}".format(path))
