
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import data
import config
import two_circles_model

import random
import time
import os, sys
import math
import argparse
from collections import OrderedDict

import numpy as np
from utils import *

class Trainer(object):

    def __init__(self, config, args):
        self.config = config
        for k, v in args.__dict__.items():
            setattr(self.config, k, v)
        setattr(self.config, 'save_dir', '{}_log'.format(self.config.dataset))
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        if torch.cuda.is_available():
            print(f'GPU: {torch.cuda.get_device_name(0)}')

        disp_str = ''
        for attr in sorted(dir(self.config), key=lambda x: len(x)):
            if not attr.startswith('__'):
                disp_str += '{} : {}\n'.format(attr, getattr(self.config, attr))
        sys.stdout.write(disp_str)
        sys.stdout.flush()

        self.labeled_loader, self.unlabeled_loader, self.unlabeled_loader2, self.dev_loader, self.special_set = data.get_two_circles_loaders(config)

        self.dis = two_circles_model.Discriminative(config).to(self.device)
        self.gen = two_circles_model.Generator(noise_size=config.noise_size, output_size=config.image_size).to(self.device)

        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=config.dis_lr, betas=(0.5, 0.999))
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=config.gen_lr, betas=(0.0, 0.999))

        self.d_criterion = nn.CrossEntropyLoss()

        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)

        self.log_path = os.path.join(self.config.save_dir, '{}.FM+PT+ENT.{}.txt'.format(self.config.dataset, self.config.suffix))
        self.logger = open(self.log_path, 'w')
        self.logger.write(disp_str)
        
        # Redirect stdout to log file
        sys.stdout = self.logger
        sys.stderr = self.logger

    def _get_vis_images(self, labels):
        labels = labels.data.cpu()
        vis_images = self.special_set.index_select(0, labels)
        return vis_images

    def _train(self, labeled=None, vis=False):
        config = self.config
        self.dis.train()
        self.gen.train()

        ##### train Dis
        lab_images, lab_labels = self.labeled_loader.next()
        lab_images, lab_labels = Variable(lab_images.to(self.device)), Variable(lab_labels.to(self.device))

        unl_images, _ = self.unlabeled_loader.next()
        unl_images = Variable(unl_images.to(self.device))

        noise = Variable(torch.Tensor(unl_images.size(0), config.noise_size).uniform_().to(self.device))
        gen_images = self.gen(noise)
        
        lab_logits = self.dis(lab_images)
        unl_logits = self.dis(unl_images)
        gen_logits = self.dis(gen_images.detach())

        # Standard classification loss (only on real classes 0 to K-1)
        lab_loss = self.d_criterion(lab_logits[:, :config.num_label], lab_labels)

        # Conditional entropy loss (only on K real classes)
        ent_loss = config.ent_weight * entropy(unl_logits[:, :config.num_label])

        # K+1 GAN loss formulation
        # For unlabeled real data: maximize probability of being in one of K real classes
        # p(y in {1..K} | x) = sum(exp(logits[:K])) / sum(exp(all_logits))
        # We want to maximize this, i.e., minimize -log of it
        # -log(p) = log_sum_exp(all) - log_sum_exp(real_K)
        unl_logsumexp_real = log_sum_exp(unl_logits[:, :config.num_label])  # sum over K real classes
        unl_logsumexp_all = log_sum_exp(unl_logits)  # sum over all K+1 classes
        true_loss = torch.mean(-unl_logsumexp_real + unl_logsumexp_all)
        
        # For generated fake data: maximize probability of being in fake class K+1
        # p(y=K+1 | x) = exp(logit_K+1) / sum(exp(all_logits))
        # -log(p) = log_sum_exp(all) - logit_K+1
        gen_logsumexp_all = log_sum_exp(gen_logits)
        fake_loss = torch.mean(-gen_logits[:, config.num_label] + gen_logsumexp_all)
        
        unl_loss = true_loss + fake_loss
         
        d_loss = lab_loss + unl_loss + ent_loss

        ##### Monitoring (train mode)
        # Unlabeled: predicted as one of K real classes (not fake)
        unl_pred = torch.argmax(unl_logits, dim=1)
        unl_acc = torch.mean((unl_pred < config.num_label).float())  # Not predicted as class K+1
        
        # Generated: predicted as fake class K+1
        gen_pred = torch.argmax(gen_logits, dim=1)
        gen_acc = torch.mean((gen_pred == config.num_label).float())  # Predicted as class K+1
        
        # Alternative metrics using probabilities
        unl_probs_real = torch.exp(unl_logsumexp_real - unl_logsumexp_all)  # p(y in {1..K})
        gen_probs_fake = torch.exp(gen_logits[:, config.num_label] - gen_logsumexp_all)  # p(y=K+1)
        max_unl_acc = torch.mean(unl_probs_real.detach().gt(0.5).float())
        max_gen_acc = torch.mean(gen_probs_fake.detach().gt(0.5).float())
        
        # Classification accuracy on labeled data (among K real classes)
        lab_pred = torch.argmax(lab_logits[:, :config.num_label], dim=1)
        lab_acc = torch.mean((lab_pred == lab_labels).float())

        self.dis_optimizer.zero_grad()
        d_loss.backward()
        self.dis_optimizer.step()

        ##### train Gen and Enc
        noise = Variable(torch.Tensor(unl_images.size(0), config.noise_size).uniform_().to(self.device))
        gen_images = self.gen(noise)

        # Feature matching loss
        unl_feat = self.dis(unl_images, feat=True)
        gen_feat = self.dis(gen_images, feat=True)
        fm_loss = torch.mean(torch.abs(torch.mean(gen_feat, 0) - torch.mean(unl_feat, 0)))

        # Entropy loss via feature pull-away term
        nsample = gen_feat.size(0)
        # normalize feature vectors with keepdim for correct broadcasting
        gen_feat_norm = gen_feat / (gen_feat.norm(p=2, dim=1, keepdim=True) + 1e-8)
        cosine = torch.mm(gen_feat_norm, gen_feat_norm.t())
        mask = Variable((torch.ones(cosine.size()) - torch.diag(torch.ones(nsample))).to(self.device))
        pt_loss = config.pt_weight * torch.sum((cosine * mask) ** 2) / (nsample * (nsample-1))

        # Generator loss
        g_loss = fm_loss + pt_loss
        
        self.gen_optimizer.zero_grad()
        g_loss.backward()
        self.gen_optimizer.step()

        monitor_dict = OrderedDict([
                       ('lab acc' , lab_acc.item()),
                       ('unl acc' , unl_acc.item()), 
                       ('gen acc' , gen_acc.item()),
                       ('max unl acc' , max_unl_acc.item()),
                       ('max gen acc' , max_gen_acc.item()),
                       ('lab loss' , lab_loss.item()),
                       ('unl loss' , unl_loss.item()),
                       ('ent loss' , ent_loss.item()),
                       ('fm loss' , fm_loss.item()),
                       ('pt loss' , pt_loss.item())
                   ])
                
        return monitor_dict

    def eval_true_fake(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()
        
        cnt = 0
        unl_acc, gen_acc, max_unl_acc, max_gen_acc = 0., 0., 0., 0.
        with torch.no_grad():
            for i, (images, _) in enumerate(data_loader.get_iter()):
                images = images.to(self.device)
                noise = torch.Tensor(images.size(0), self.config.noise_size).uniform_().to(self.device)

                unl_feat = self.dis(images, feat=True)
                gen_feat = self.dis(self.gen(noise), feat=True)

                unl_logits = self.dis.out_net(unl_feat)
                gen_logits = self.dis.out_net(gen_feat)

                ##### Monitoring (eval mode)
                # Unlabeled real: predicted as one of K real classes
                unl_pred = torch.argmax(unl_logits, dim=1)
                unl_acc += torch.mean((unl_pred < self.config.num_label).float()).item()
                
                # Generated: predicted as fake class K+1
                gen_pred = torch.argmax(gen_logits, dim=1)
                gen_acc += torch.mean((gen_pred == self.config.num_label).float()).item()
                
                # Alternative metrics using probabilities
                unl_logsumexp_real = log_sum_exp(unl_logits[:, :self.config.num_label])
                unl_logsumexp_all = log_sum_exp(unl_logits)
                gen_logsumexp_all = log_sum_exp(gen_logits)
                unl_probs_real = torch.exp(unl_logsumexp_real - unl_logsumexp_all)
                gen_probs_fake = torch.exp(gen_logits[:, self.config.num_label] - gen_logsumexp_all)
                max_unl_acc += torch.mean(unl_probs_real.gt(0.5).float()).item()
                max_gen_acc += torch.mean(gen_probs_fake.gt(0.5).float()).item()

                cnt += 1
                if max_batch is not None and i >= max_batch - 1: break

        return unl_acc / cnt, gen_acc / cnt, max_unl_acc / cnt, max_gen_acc / cnt

    def eval(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()

        loss, incorrect, cnt = 0, 0, 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader.get_iter()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.dis(images)
                # Only use real class logits for classification
                real_logits = logits[:, :self.config.num_label]
                loss += self.d_criterion(real_logits, labels).item()
                cnt += 1
                pred_classes = torch.argmax(real_logits, dim=1)
                incorrect += torch.ne(pred_classes, labels).sum().item()
                if max_batch is not None and i >= max_batch - 1: break
        return loss / cnt, incorrect


    def visualize(self):
        self.gen.eval()
        self.dis.eval()

        # Generate samples
        vis_size = 1000
        noise = Variable(torch.Tensor(vis_size, self.config.noise_size).uniform_().to(self.device))
        gen_samples = self.gen(noise).data.cpu().numpy()

        # Get real samples for comparison
        real_samples = []
        for i, (images, labels) in enumerate(self.dev_loader.get_iter()):
            real_samples.append(images.numpy())
            if len(real_samples) * images.size(0) >= vis_size:
                break
        real_samples = np.concatenate(real_samples, axis=0)[:vis_size]

        # Create simple visualization (2 subplots: real vs generated)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot real data
        axes[0].scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, s=10, c='gray')
        axes[0].set_title('Real Data')
        axes[0].set_xlim(-1.5, 1.5)
        axes[0].set_ylim(-1.5, 1.5)
        axes[0].grid(True, alpha=0.3)
        
        # Plot generated data
        axes[1].scatter(gen_samples[:, 0], gen_samples[:, 1], alpha=0.5, s=10, c='red')
        axes[1].set_title('Generated Data')
        axes[1].set_xlim(-1.5, 1.5)
        axes[1].set_ylim(-1.5, 1.5)
        axes[1].grid(True, alpha=0.3)
        
        save_path = os.path.join(self.config.save_dir, '{}.FM+PT+Ent.{}.png'.format(self.config.dataset, self.config.suffix))
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
    
    def visualize_detailed(self, iter_num):
        """Detailed visualization showing labeled, unlabeled, and generated data with predictions"""
        self.gen.eval()
        self.dis.eval()

        # Generate samples
        gen_size = 500
        noise = Variable(torch.Tensor(gen_size, self.config.noise_size).uniform_().to(self.device))
        gen_samples = self.gen(noise).data.cpu().numpy()

        # Get labeled samples (both classes) with predictions
        labeled_samples_inner = []  # class 0 (inner circle)
        labeled_samples_outer = []  # class 1 (outer circle)
        labeled_images_list = []
        labeled_true_labels = []
        
        for i, (images, labels) in enumerate(self.labeled_loader.get_iter()):
            for j in range(images.size(0)):
                if labels[j] == 0:
                    labeled_samples_inner.append(images[j].numpy())
                else:
                    labeled_samples_outer.append(images[j].numpy())
                labeled_images_list.append(images[j].numpy())
                labeled_true_labels.append(labels[j].item())
        
        labeled_samples_inner = np.array(labeled_samples_inner) if labeled_samples_inner else np.empty((0, 2))
        labeled_samples_outer = np.array(labeled_samples_outer) if labeled_samples_outer else np.empty((0, 2))
        
        # Get predictions for labeled data
        if len(labeled_images_list) > 0:
            labeled_images_tensor = torch.stack([torch.from_numpy(img) for img in labeled_images_list]).to(self.device)
            with torch.no_grad():
                labeled_logits = self.dis(labeled_images_tensor)
                # Use only real class logits for classification
                labeled_pred_classes = torch.argmax(labeled_logits[:, :self.config.num_label], dim=1).cpu().numpy()
                # Also check if any are classified as fake
                labeled_full_pred = torch.argmax(labeled_logits, dim=1).cpu().numpy()
            labeled_true_labels = np.array(labeled_true_labels)
            labeled_accuracy = np.mean(labeled_pred_classes == labeled_true_labels)
        else:
            labeled_accuracy = 0.0
            labeled_pred_classes = np.array([])

        # Get unlabeled samples with predictions
        unlabeled_samples = []
        unlabeled_images_list = []
        for i, (images, _) in enumerate(self.unlabeled_loader.get_iter()):
            unlabeled_samples.append(images.numpy())
            unlabeled_images_list.append(images.numpy())
            if len(unlabeled_samples) * images.size(0) >= 500:
                break
        unlabeled_samples = np.concatenate(unlabeled_samples, axis=0)[:500]
        unlabeled_images_cat = np.concatenate(unlabeled_images_list, axis=0)[:500]
        
        # Get predictions for unlabeled data using K+1 class probabilities
        unlabeled_images_tensor = torch.from_numpy(unlabeled_images_cat).to(self.device)
        with torch.no_grad():
            unlabeled_logits = self.dis(unlabeled_images_tensor)
            # Predicted class (0, 1, or 2=fake)
            unlabeled_pred_classes = torch.argmax(unlabeled_logits, dim=1).cpu().numpy()
            # For classification among K real classes (when not predicted as fake)
            unlabeled_pred_real_classes = torch.argmax(unlabeled_logits[:, :self.config.num_label], dim=1).cpu().numpy()
        
        # Separate unlabeled by predicted class
        unlabeled_pred_fake = unlabeled_samples[unlabeled_pred_classes == self.config.num_label]  # Predicted as fake (wrong!)
        unlabeled_not_fake_mask = unlabeled_pred_classes < self.config.num_label
        unlabeled_pred_inner = unlabeled_samples[unlabeled_not_fake_mask & (unlabeled_pred_real_classes == 0)]
        unlabeled_pred_outer = unlabeled_samples[unlabeled_not_fake_mask & (unlabeled_pred_real_classes == 1)]
        
        # Get predictions for generated data using K+1 class
        gen_images_tensor = torch.from_numpy(gen_samples).float().to(self.device)
        with torch.no_grad():
            gen_logits = self.dis(gen_images_tensor)
            # Predicted class (0, 1, or 2=fake)
            gen_pred_classes = torch.argmax(gen_logits, dim=1).cpu().numpy()
            # For classification among K real classes (when wrongly predicted as real)
            gen_pred_real_classes = torch.argmax(gen_logits[:, :self.config.num_label], dim=1).cpu().numpy()
        
        # Separate generated: correctly predicted as fake vs wrongly predicted as real
        gen_pred_fake_correct = gen_samples[gen_pred_classes == self.config.num_label]  # Correctly predicted as fake!
        gen_wrongly_real_mask = gen_pred_classes < self.config.num_label
        gen_pred_inner = gen_samples[gen_wrongly_real_mask & (gen_pred_real_classes == 0)]  # Wrong: predicted as inner
        gen_pred_outer = gen_samples[gen_wrongly_real_mask & (gen_pred_real_classes == 1)]  # Wrong: predicted as outer

        # Create detailed visualization with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        
        # Left plot: Ground truth labels
        ax = axes[0]
        
        # Plot unlabeled data (gray)
        ax.scatter(unlabeled_samples[:, 0], unlabeled_samples[:, 1], 
                  alpha=0.3, s=20, c='gray', label='Unlabeled', edgecolors='none')
        
        # Plot labeled data - inner circle (blue)
        if len(labeled_samples_inner) > 0:
            ax.scatter(labeled_samples_inner[:, 0], labeled_samples_inner[:, 1], 
                      alpha=0.8, s=60, c='blue', label='Labeled (Inner)', edgecolors='black', linewidth=0.8, marker='o')
        
        # Plot labeled data - outer circle (orange)
        if len(labeled_samples_outer) > 0:
            ax.scatter(labeled_samples_outer[:, 0], labeled_samples_outer[:, 1], 
                      alpha=0.8, s=60, c='orange', label='Labeled (Outer)', edgecolors='black', linewidth=0.8, marker='o')
        
        # Plot generated data (red)
        ax.scatter(gen_samples[:, 0], gen_samples[:, 1], 
                  alpha=0.5, s=15, c='red', label='Generated', edgecolors='none')
        
        ax.set_title(f'Ground Truth - Iteration {iter_num}', fontsize=14, fontweight='bold')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_aspect('equal')
        
        # Right plot: Discriminator predictions
        ax = axes[1]
        
        # Plot unlabeled data by predicted class
        if len(unlabeled_pred_inner) > 0:
            ax.scatter(unlabeled_pred_inner[:, 0], unlabeled_pred_inner[:, 1], 
                      alpha=0.3, s=20, c='lightblue', label='Unlabeled (Pred: Inner)', edgecolors='none')
        if len(unlabeled_pred_outer) > 0:
            ax.scatter(unlabeled_pred_outer[:, 0], unlabeled_pred_outer[:, 1], 
                      alpha=0.3, s=20, c='wheat', label='Unlabeled (Pred: Outer)', edgecolors='none')
        if len(unlabeled_pred_fake) > 0:
            ax.scatter(unlabeled_pred_fake[:, 0], unlabeled_pred_fake[:, 1], 
                      alpha=0.5, s=25, c='purple', label='Unlabeled (Pred: Fake)', edgecolors='black', linewidth=0.3, marker='s')
        
        # Plot labeled data with correct/incorrect predictions
        if len(labeled_images_list) > 0:
            labeled_samples_arr = np.array(labeled_images_list)
            correct_mask = labeled_pred_classes == labeled_true_labels
            incorrect_mask = ~correct_mask
            
            # Correctly classified
            if np.any(correct_mask):
                ax.scatter(labeled_samples_arr[correct_mask, 0], labeled_samples_arr[correct_mask, 1], 
                          alpha=0.8, s=60, c='green', label='Labeled (Correct)', 
                          edgecolors='black', linewidth=0.8, marker='o')
            
            # Misclassified
            if np.any(incorrect_mask):
                ax.scatter(labeled_samples_arr[incorrect_mask, 0], labeled_samples_arr[incorrect_mask, 1], 
                          alpha=0.8, s=80, c='red', label='Labeled (Wrong)', 
                          edgecolors='black', linewidth=1.2, marker='X')
        
        # Plot generated data by predicted class
        if len(gen_pred_inner) > 0:
            ax.scatter(gen_pred_inner[:, 0], gen_pred_inner[:, 1], 
                      alpha=0.5, s=15, c='darkblue', label='Generated (Pred: Inner - Wrong!)', edgecolors='none')
        if len(gen_pred_outer) > 0:
            ax.scatter(gen_pred_outer[:, 0], gen_pred_outer[:, 1], 
                      alpha=0.5, s=15, c='darkorange', label='Generated (Pred: Outer - Wrong!)', edgecolors='none')
        if len(gen_pred_fake_correct) > 0:
            ax.scatter(gen_pred_fake_correct[:, 0], gen_pred_fake_correct[:, 1], 
                      alpha=0.7, s=20, c='lime', label='Generated (Pred: Fake/Class K+1 - Correct!)', edgecolors='black', linewidth=0.3, marker='*')
        
        # Calculate metrics
        fake_detection_rate = len(gen_pred_fake_correct) / len(gen_samples) * 100 if len(gen_samples) > 0 else 0
        unlabeled_as_fake_rate = len(unlabeled_pred_fake) / len(unlabeled_samples) * 100 if len(unlabeled_samples) > 0 else 0
        
        title_text = f'Discriminator Predictions (K+1 Classes: Inner/Outer/Fake) - Iteration {iter_num}\n'
        title_text += f'Labeled Acc: {labeled_accuracy*100:.1f}%'
        if len(labeled_images_list) > 0:
            title_text += f' ({int(labeled_accuracy * len(labeled_images_list))}/{len(labeled_images_list)})'
        title_text += f' | Unlabeled Acc: {100-unlabeled_as_fake_rate:.1f}% (real→real)'
        title_text += f' | Fake Detection: {fake_detection_rate:.1f}% (fake→K+1)'
        
        ax.set_title(title_text, fontsize=14, fontweight='bold')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_aspect('equal')
        
        save_path = os.path.join(self.config.save_dir, '{}.detailed.iter_{}.png'.format(self.config.dataset, iter_num))
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        # Log accuracy to console/file
        print(f'\n[Visualization {iter_num}] Labeled Data Accuracy: {labeled_accuracy*100:.2f}%')

    def param_init(self):
        # Simple initialization for 2D data
        pass

    def train(self):
        config = self.config
        self.param_init()

        self.iter_cnt = 0
        iter, min_dev_incorrect = 0, 1e6
        monitor = OrderedDict()
        
        batch_per_epoch = int((len(self.unlabeled_loader) + config.train_batch_size - 1) / config.train_batch_size)
        min_lr = 0.0
        
        while True:

            if iter % batch_per_epoch == 0:
                epoch = iter // batch_per_epoch
                if epoch >= config.max_epochs:
                    break
                epoch_ratio = float(epoch) / float(config.max_epochs)
                # use another outer max to prevent any float computation precision problem
                self.dis_optimizer.param_groups[0]['lr'] = max(min_lr, config.dis_lr * min(3. * (1. - epoch_ratio), 1.))
                self.gen_optimizer.param_groups[0]['lr'] = max(min_lr, config.gen_lr * min(3. * (1. - epoch_ratio), 1.))

            iter_vals = self._train()

            for k, v in iter_vals.items():
                if k not in monitor:
                    monitor[k] = 0.
                monitor[k] += v

            if iter % config.vis_period == 0:
                self.visualize()
            
            # Detailed visualization with all data types
            if hasattr(config, 'plot_period') and iter % config.plot_period == 0:
                self.visualize_detailed(iter)

            if iter % config.eval_period == 0:
                train_loss, train_incorrect = self.eval(self.labeled_loader)
                dev_loss, dev_incorrect = self.eval(self.dev_loader)

                unl_acc, gen_acc, max_unl_acc, max_gen_acc = self.eval_true_fake(self.dev_loader, 10)

                train_incorrect /= 1.0 * len(self.labeled_loader)
                dev_incorrect /= 1.0 * len(self.dev_loader)
                min_dev_incorrect = min(min_dev_incorrect, dev_incorrect)

                disp_str = '#{}\ttrain: {:.4f}, {:.4f} | dev: {:.4f}, {:.4f} | best: {:.4f}'.format(
                    iter, train_loss, train_incorrect, dev_loss, dev_incorrect, min_dev_incorrect)
                for k, v in monitor.items():
                    disp_str += ' | {}: {:.4f}'.format(k, v / config.eval_period)
                
                disp_str += ' | [Eval] unl acc: {:.4f}, gen acc: {:.4f}, max unl acc: {:.4f}, max gen acc: {:.4f}'.format(unl_acc, gen_acc, max_unl_acc, max_gen_acc)
                disp_str += ' | lr: {:.5f}'.format(self.dis_optimizer.param_groups[0]['lr'])
                disp_str += '\n'

                monitor = OrderedDict()

                self.logger.write(disp_str)
                sys.stdout.write(disp_str)
                sys.stdout.flush()

            iter += 1
            self.iter_cnt += 1

        # Restore stdout before closing
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.logger.close()
        print('Training completed! Check log file at:', self.log_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='two_circles_trainer.py')
    parser.add_argument('-suffix', default='run0', type=str, help="Suffix added to the save images.")

    args = parser.parse_args()

    trainer = Trainer(config.two_circles_config(), args)
    trainer.train()
