import numpy as np
import os
import time
import datetime
from copy import deepcopy

import torch
from torch import nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from utils import *
from models import Generator, Discriminator
from data.sparse_molecular_dataset import SparseMolecularDataset

from sklearn.metrics import average_precision_score

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config):
        """Initialize configurations."""

        self.verbose = False

        # Data loader.
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)

        # Model configurations.
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.post_method = config.post_method

        self.metric = 'validity,sas'

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # class vars
        self.disc_label = torch.autograd.Variable(torch.FloatTensor(self.batch_size).to(self.device))
        self.aux_label = torch.autograd.Variable(torch.FloatTensor(self.batch_size).to(self.device))

        self.real_label = 1
        self.fake_label = 0
        
        self.writer = SummaryWriter(config.summary_dir)

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.z_dim,
                           self.data.vertexes,
                           self.data.bond_num_types,
                           self.data.atom_num_types,
                           self.dropout)
        self.D = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout)
        self.V = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout)

        """Create optimizers for generator and discriminator"""
        self.g_optimizer = torch.optim.Adam(list(self.G.parameters())+list(self.V.parameters()),
                                            self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(),
                                            self.d_lr, [self.beta1, self.beta2])

        """Define loss functions for generator, discriminator, and aux"""
        self.disc_loss = nn.BCELoss() # added
        self.aux_loss = nn.MSELoss() # added

        """Print networks"""
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.V, 'V')

        self.G.to(self.device)
        self.D.to(self.device)
        self.V.to(self.device)
        self.disc_loss.to(self.device)
        self.aux_loss.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        # Convert the range from [-1, 1] to [0, 1].
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size())+[dim]).to(self.device)
        out.scatter_(len(out.size())-1,labels.unsqueeze(-1), 1.)
        return out

    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    def add_logP_to_z(self, z, logP):
        z[:,0]=logP
        return z

    def postprocess(self, inputs, method, temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    def reward(self, mols):
        if self.verbose: 
            print("def reward: ")
        rr = 1.
        for m in ('sas,qed,unique' if self.metric == 'all' else self.metric).split(','):
            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
                if self.verbose:
                    print("MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True): ")
                    print(MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True))
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
                if self.verbose:
                    print("MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True): ")
                    print(MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True))
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
                if self.verbose:
                    print("MolecularMetrics.unique_scores(mols): ")
                    print(MolecularMetrics.unique_scores(mols))
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))
        return rr.reshape(-1, 1)

    def train(self):

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            if (i+1) % self.log_step == 0:
                mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
                z = self.sample_z(a.shape[0])
                real_logPs = MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
                fake_logPs = np.random.normal(0,1, size=a.shape[0])
                label_real_disc = deepcopy(self.disc_label.data.resize_(a.shape[0]).fill_(self.real_label))
                label_fake_disc = deepcopy(self.disc_label.data.resize_(a.shape[0]).fill_(self.fake_label))
                print('[Valid]', '')
            else:
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
                z = self.sample_z(self.batch_size)
                real_logPs = MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
                fake_logPs = np.random.normal(0,1, size=(self.batch_size))
                label_real_disc = deepcopy(self.disc_label.data.resize_(self.batch_size).fill_(self.real_label))
                label_fake_disc = deepcopy(self.disc_label.data.resize_(self.batch_size).fill_(self.fake_label))

            z = self.add_logP_to_z(z, fake_logPs)

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            a = torch.from_numpy(a).to(self.device).long()            # Adjacency.
            x = torch.from_numpy(x).to(self.device).long()            # Nodes: represented as atomic number
            a_tensor = self.label2onehot(a, self.b_dim) # 16x9x9x5
            x_tensor = self.label2onehot(x, self.m_dim) # 16x9x5
            z = torch.from_numpy(z).to(self.device).float() # 16x8 # batchx8

            real_logPs = torch.from_numpy(real_logPs).to(self.device).float()
            fake_logPs = torch.from_numpy(fake_logPs).to(self.device).float()

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # TRAIN W/REAL
            logits_real_disc, logits_real_aux, features_real = self.D(a_tensor, None, x_tensor)
            disc_loss_real = self.disc_loss(logits_real_disc, label_real_disc)
            aux_loss_real = self.aux_loss(real_logPs, logits_real_aux)
            disc_loss_real_all = disc_loss_real + aux_loss_real 
            disc_loss_real_all.backward() 

            # TRAIN W/FAKE
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax believe this is where categorical sampling occurs
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            logits_fake_disc, logits_fake_aux, features_fake = self.D(edges_hat, None, nodes_hat)
            disc_loss_fake = self.disc_loss(logits_fake_disc, label_fake_disc)
            aux_loss_fake = self.aux_loss(fake_logPs, logits_fake_aux)
            disc_loss_fake_all = disc_loss_fake + aux_loss_fake 
            disc_loss_fake_all.backward() 

            # Compute loss for gradient penalty.
            eps = torch.rand(logits_real_disc.size(0),1,1,1).to(self.device)
            x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
            x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
            #grad0, grad1, _ = self.D(x_int0, None, x_int1) # TODO ASAP figure out what to do here
            grad0, _, grad1 = self.D(x_int0, None, x_int1) # TODO ASAP figure out what to do here
            d_loss_gp = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)

            # Backward and optimize D
            disc_loss_all = disc_loss_real_all + disc_loss_fake_all + self.lambda_gp * d_loss_gp
            self.reset_grad()
            self.d_optimizer.step()

            # compute classification accuracy of discriminator
            if (i+1) % self.log_step == 0:
                """
                label_disc = label_real_disc+label_fake_disc
                logPs = real_logPs+fake_logPs
                logits_aux = logits_real_aux+logits_fake_aux
                logits_aux = logits_aux.view(logits_aux.shape[0])
                """
                self.writer.add_scalar("train_disc_loss",
                                       disc_loss_all,
                                       i)
                self.writer.add_scalar("train_disc_aux_loss_fake",
                                       torch.mean(aux_loss_fake).detach().numpy(),
                                       i)
                self.writer.add_scalar("train_disc_aux_loss_real",
                                       torch.mean(aux_loss_real).detach().numpy(),
                                       i)
            # Logging
            loss = {}
            loss['D/disc_loss_real_all'] = disc_loss_real_all.item()
            loss['D/disc_loss_fake_all'] = disc_loss_fake_all.item()
            loss['D/disc_loss_all'] = disc_loss_all.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Z-to-target
                edges_logits, nodes_logits = self.G(z)
                # Postprocess with Gumbel softmax
                (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
                logits_fake_disc, logits_fake_aux, features_fake = self.D(edges_hat, None, nodes_hat)
                g_loss_fake = - torch.mean(logits_fake_disc)

                # Real Reward TODO cut this out
                if self.verbose: 
                    print("real reward")
                rewardR = torch.from_numpy(self.reward(mols)).to(self.device)
                # Fake Reward
                if self.verbose: 
                    print("fake reward")
                (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
                edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
                mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                        for e_, n_ in zip(edges_hard, nodes_hard)]
                rewardF = torch.from_numpy(self.reward(mols)).to(self.device)

                # Value loss
                value_logit_real, _, _ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
                value_logit_fake, _, _ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)
                g_loss_value = torch.mean((value_logit_real - rewardR) ** 2 +
                                          (value_logit_fake - rewardF) ** 2)

                rl_loss = -value_logit_fake
                f_loss = (torch.mean(features_real, 0) - torch.mean(features_fake, 0)) ** 2

                # Backward and optimize.
                g_loss = g_loss_fake + g_loss_value
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_value'] = g_loss_value.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)

                # Log update
                m0, m1 = all_scores(mols, self.data, norm=True)     # 'mols' is output of Fake Reward
                m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
                m0.update(m1)
                loss.update(m0)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.V.state_dict(), V_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    """
    def test(self):
        # Load the trained generator.
        self.restore_model(self.test_iters)

        with torch.no_grad():
            mols, _, _, a, x, _, _, _, _ = self.data.next_test_batch()
            z = self.sample_z(a.shape[0])

            # Z-to-target
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
            g_loss_fake = - torch.mean(logits_fake)

            # Fake Reward
            (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
            edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
            mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                    for e_, n_ in zip(edges_hard, nodes_hard)]

            # Log update
            m0, m1 = all_scores(mols, self.data, norm=True)     # 'mols' is output of Fake Reward
            m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
            m0.update(m1)
            for tag, value in m0.items():
                log += ", {}: {:.4f}".format(tag, value)
    """
