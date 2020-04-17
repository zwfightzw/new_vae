import os
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.optim as optim
from RNN_cell import GRUCell, LSTMCell
import numpy as np
from torch.distributions import Normal, kl_divergence
import datetime
import dateutil.tz
import argparse

def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()

class Sprites(torch.utils.data.Dataset):
    def __init__(self, path, size):
        self.path = path
        self.length = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.load(self.path + '/%d.sprite' % (idx + 1))


class FullQDisentangledVAE(nn.Module):
    def __init__(self, frames, z_dim, conv_dim, hidden_dim, device):
        super(FullQDisentangledVAE, self).__init__()
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.z_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                              bidirectional=True, batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)

        self.z_post_fwd = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.z_post_out = nn.Linear(self.hidden_dim, self.z_dim*2)

        self.z_prior_fwd = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.z_prior_out = nn.Linear(self.hidden_dim, self.z_dim * 2)

        #self.z_to_z_fwd = LSTMCell(input_size=self.z_dim, hidden_size=self.hidden_dim).to(device)
        self.z_to_z_fwd = GRUCell(input_size=self.z_dim, hidden_size=self.hidden_dim).to(device)

        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.drop2 = nn.Dropout2d(0.4)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.drop3 = nn.Dropout2d(0.4)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout2d(0.4)
        self.conv_fc = nn.Linear(4 * 4 * 256, self.conv_dim)  # 4*4 is size 256 is channels
        self.drop_fc = nn.Dropout(0.4)
        self.bnf = nn.BatchNorm1d(self.conv_dim)

        self.deconv_fc = nn.Linear(self.z_dim, 4 * 4 * 256)  # 4*4 is size 256 is channels
        self.deconv_bnf = nn.BatchNorm1d(4 * 4 * 256)
        self.drop_fc_deconv = nn.Dropout(0.4)
        self.deconv4 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.dbn4 = nn.BatchNorm2d(256)
        self.drop4_deconv = nn.Dropout2d(0.4)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.dbn3 = nn.BatchNorm2d(256)
        self.drop3_deconv = nn.Dropout2d(0.4)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.dbn2 = nn.BatchNorm2d(256)
        self.drop2_deconv = nn.Dropout2d(0.4)
        self.deconv1 = nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        nonlinearity='relu')  # Change nonlinearity to 'leaky_relu' if you switch
        nn.init.xavier_normal_(self.deconv1.weight, nn.init.calculate_gain('tanh'))

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def encode_frames(self, x):
        x = x.view(-1, 3, 64, 64)  # Batchwise stack the 8 images for applying convolutions parallely
        x = F.leaky_relu(self.conv1(x), 0.1)  # Remove batchnorm, the encoder must learn the data distribution
        x = self.drop2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        x = self.drop3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1))
        x = self.drop4(F.leaky_relu(self.bn4(self.conv4(x)), 0.1))
        x = x.view(-1, 4 * 4 * 256)  # 4*4 is size 256 is channels
        x = self.drop_fc(F.leaky_relu(self.bnf(self.conv_fc(x)), 0.1))
        x = x.view(-1, self.frames, self.conv_dim)
        return x

    def decode_frames(self, zf):
        x = zf.view(-1, self.z_dim)  # For batchnorm1D to work, the frames should be stacked batchwise
        x = self.drop_fc_deconv(F.leaky_relu(self.deconv_bnf(self.deconv_fc(x)), 0.1))
        x = x.view(-1, 256, 4, 4)  # The 8 frames are stacked batchwise
        x = self.drop4_deconv(F.leaky_relu(self.dbn4(self.deconv4(x)), 0.1))
        x = self.drop3_deconv(F.leaky_relu(self.dbn3(self.deconv3(x)), 0.1))
        x = self.drop2_deconv(F.leaky_relu(self.dbn2(self.deconv2(x)), 0.1))
        x = torch.tanh(self.deconv1(
            x))  # Images are normalized to -1,1 range hence use tanh. Remove batchnorm because it should fit the final distribution
        return x.view(-1, self.frames, 3, 64, 64)  # Convert the stacked batches back into frames. Images are 64*64*3

    def encode_z(self, x):
        lstm_out, _ = self.z_lstm(x)
        lstm_out, _ = self.z_rnn(lstm_out)
        lstm_out = self.z_post_fwd(lstm_out)

        zt_obs_list = []
        batch_size = lstm_out.shape[0]
        seq_size = lstm_out.shape[1]

        zt_1 = torch.zeros(batch_size, self.z_dim).to(device)
        z_state_hx = zt_1.new_zeros(batch_size, self.hidden_dim)
        z_state_cx = zt_1.new_zeros(batch_size, self.hidden_dim)

        z_post_mean_list = []
        z_post_lar_list = []
        z_prior_mean_list = []
        z_prior_lar_list = []

        for t in range(0, seq_size):
            # posterior over ct, q(ct|ot,ft)
            z_post_out = self.z_post_out(lstm_out[:,t])
            zt_post_mean = z_post_out[:,:self.z_dim]
            zt_post_lar =z_post_out[:,self.z_dim:]

            z_post_mean_list.append(zt_post_mean)
            z_post_lar_list.append(zt_post_lar)
            z_post_sample = self.reparameterize(zt_post_mean, zt_post_lar, self.training)

            # p(xt|zt)
            zt_obs_list.append(z_post_sample)

            # prior over ct of each block, ct_i~p(ct_i|zt-1_i)
            #z_state_hx,  z_state_cx = self.z_to_z_fwd(zt_1, (z_state_hx, z_state_cx))
            z_state_hx = self.z_to_z_fwd(zt_1, z_state_hx)
            z_prior_fwd = self.z_prior_fwd(z_state_hx)
            z_prior_fwd = self.z_prior_out(z_prior_fwd)

            z_fwd_latent_mean = z_prior_fwd[:,:self.z_dim]
            z_fwd_latent_lar = z_prior_fwd[:,self.z_dim:]

            z_prior_mean_list.append(z_fwd_latent_mean)
            z_prior_lar_list.append(z_fwd_latent_lar)

            zt_1= self.reparameterize(z_fwd_latent_mean, z_fwd_latent_lar, self.training)

        zt_obs_list = torch.stack(zt_obs_list, dim=1)
        z_post_mean_list = torch.stack(z_post_mean_list, dim=1)
        z_post_lar_list = torch.stack(z_post_lar_list, dim=1)
        z_prior_mean_list = torch.stack(z_prior_mean_list, dim=1)
        z_prior_lar_list = torch.stack(z_prior_lar_list, dim=1)

        return z_post_mean_list, z_post_lar_list, z_prior_mean_list, z_prior_lar_list, zt_obs_list

    def forward(self, x):
        conv_x = self.encode_frames(x)

        post_zt_mean, post_zt_lar, prior_zt_mean, prior_zt_lar, z = self.encode_z(conv_x)
        recon_x = self.decode_frames(z)
        return post_zt_mean, post_zt_lar, prior_zt_mean, prior_zt_lar, z, recon_x

def loss_fn(original_seq, recon_seq, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar):
    batch_size = original_seq.shape[0]
    mse = F.mse_loss(recon_seq,original_seq,reduction='sum')
    # compute kl related to states, kl(q(ct|ot,ft)||p(ct|zt-1))
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * torch.sum(
        z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)
    return (mse + kld_z) / batch_size,  kld_z / batch_size

class Trainer(object):
    def __init__(self, model, device, train, test, trainloader, testloader, epochs, batch_size, learning_rate, nsamples,
                 sample_path, recon_path, checkpoints, log_path, grad_clip):
        self.trainloader = trainloader
        self.train = train
        self.test = test
        self.testloader = testloader
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.grad_clip = grad_clip
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.samples = nsamples
        self.sample_path = sample_path
        self.recon_path = recon_path
        self.log_path = log_path
        self.epoch_losses = []

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.epoch_losses},
            self.checkpoints)

    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def sample_frames(self, epoch):
        with torch.no_grad():

            zt_dec = []
            '''
            prior_z0 = torch.distributions.Normal(torch.zeros(self.model.z_dim).to(self.device),
                                                  torch.ones(self.model.z_dim).to(self.device))

            zt_1 = [prior_z0.rsample() for i in range(self.samples)]
            zt_1 = torch.stack(zt_1, dim=0)
            zt_dec.append(zt_1)
            '''
            zt_1 = torch.zeros(self.samples, self.model.z_dim).to(device)
            z_state_hx = zt_1.new_zeros(self.samples, self.model.hidden_dim)
            z_state_cx = zt_1.new_zeros(self.samples, self.model.hidden_dim)

            for t in range(0, 8):

                # prior over ct of each block, ct_i~p(ct_i|zt-1_i)
                #z_state_hx, z_state_cx = self.model.z_to_z_fwd(zt_1, (z_state_hx, z_state_cx))
                z_state_hx = self.model.z_to_z_fwd(zt_1, z_state_hx)

                z_prior_fwd = self.model.z_prior_fwd(z_state_hx)
                z_prior_fwd = self.model.z_prior_out(z_prior_fwd)

                z_fwd_latent_mean = z_prior_fwd[:, :self.model.z_dim]
                z_fwd_latent_lar = z_prior_fwd[:, self.model.z_dim:]

                zt = self.model.reparameterize(z_fwd_latent_mean, z_fwd_latent_lar, False)
                zt_dec.append(zt)
                zt_1 = zt

            zt_dec = torch.stack(zt_dec, dim=1)
            recon_x = self.model.decode_frames(zt_dec)
            recon_x = recon_x.view(16, 3, 64, 64)
            torchvision.utils.save_image(recon_x, '%s/epoch%d.png' % (self.sample_path, epoch))

    def recon_frame(self, epoch, original):
        with torch.no_grad():
            _, _, _, _,_, recon = self.model(original)
            image = torch.cat((original, recon), dim=0)
            print(image.shape)
            image = image.view(16, 3, 64, 64)
            torchvision.utils.save_image(image, '%s/epoch%d.png' % (self.recon_path, epoch))

    def train_model(self):

        self.model.eval()
        self.sample_frames(0 + 1)
        sample = self.test[int(torch.randint(0, len(self.test), (1,)).item())]
        sample = torch.unsqueeze(sample, 0)
        sample = sample.to(self.device)
        self.recon_frame(0 + 1, sample)

        self.model.train()

        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            kl_loss = []
            write_log("Running Epoch : {}".format(epoch + 1), self.log_path)
            for i, data in enumerate(self.trainloader, 1):
                data = data.to(device)
                self.optimizer.zero_grad()
                post_zt_mean, post_zt_lar, prior_zt_mean, prior_zt_lar, z, recon_x = self.model(data)
                loss, kl = loss_fn(data, recon_x, post_zt_mean, post_zt_lar, prior_zt_mean, prior_zt_lar)
                loss.backward()
                if self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                losses.append(loss.item())
                kl_loss.append(kl.item())
            meanloss = np.mean(losses)
            klloss = np.mean(kl_loss)
            self.epoch_losses.append(meanloss,klloss)
            write_log("Epoch {} : Average Loss: {}, kl loss: {}".format(epoch + 1, meanloss, klloss), self.log_path)
            #self.save_checkpoint(epoch)
            self.model.eval()
            self.sample_frames(epoch + 1)
            sample = self.test[int(torch.randint(0, len(self.test), (1,)).item())]
            sample = torch.unsqueeze(sample, 0)
            sample = sample.to(self.device)
            self.recon_frame(epoch + 1, sample)
            #self.style_transfer(epoch + 1)
            self.model.train()
        print("Training is complete")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="vanilla_vae")
    parser.add_argument('--seed', type=int, default=111)
    # method
    parser.add_argument('--method', type=str, default='Vanilla')
    # dataset
    parser.add_argument('--dset_name', type=str, default='moving_mnist')
    # state size
    parser.add_argument('--z-dim', type=int, default=36)
    parser.add_argument('--hidden-dim', type=int, default=384)
    parser.add_argument('--conv-dim', type=int, default=1024)
    # data size
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--frame-size', type=int, default=8)
    parser.add_argument('--nsamples', type=int, default=2)

    # optimization
    parser.add_argument('--learn-rate', type=float, default=0.0002)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--gpu_id', type=int, default=1)

    FLAGS = parser.parse_args()
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    device = torch.device('cuda:%d'%(FLAGS.gpu_id) if torch.cuda.is_available() else 'cpu')

    vae = FullQDisentangledVAE(frames=FLAGS.frame_size, z_dim=FLAGS.z_dim, hidden_dim=FLAGS.hidden_dim, conv_dim=FLAGS.conv_dim, device=device)
    print(vae)
    sprites_train = Sprites('./dataset/lpc-dataset/train/', 6687)
    sprites_test = Sprites('./dataset/lpc-dataset/test/', 873)
    starttime = datetime.datetime.now()
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    time_dir = now.strftime('%Y_%m_%d_%H_%M_%S')
    base_path = './%s/%s'%(FLAGS.method, time_dir)
    model_path = '%s/model' % (base_path)
    log_recon = '%s/recon' % (base_path)
    log_sample = '%s/sample' % (base_path)
    log_path = '%s/log_info.txt' % (base_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_recon):
        os.makedirs(log_recon)
    if not os.path.exists(log_sample):
        os.makedirs(log_sample)

    write_log(FLAGS, log_path)
    write_log(vae, log_path)

    trainloader = torch.utils.data.DataLoader(sprites_train, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(sprites_test, batch_size=1, shuffle=True, num_workers=4)

    trainer = Trainer(vae, device, sprites_train, sprites_test, trainloader, testloader, epochs=FLAGS.max_epochs, batch_size=FLAGS.batch_size,
                      learning_rate=FLAGS.learn_rate, checkpoints='%s/%s-disentangled-vae.model'%(model_path, FLAGS.method), nsamples=FLAGS.nsamples,
                      sample_path=log_sample,
                      recon_path=log_recon, log_path=log_path, grad_clip=FLAGS.grad_clip)
    #trainer.load_checkpoint()
    trainer.train_model()
    endtime = datetime.datetime.now()
    seconds = (endtime - starttime).seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    second = (seconds % 3600) % 60
    print((endtime - starttime))
    timeStr = "running time: " + str(hours) + 'hours' + str(minutes) + 'minutes' + str(second) + "second"
    write_log(timeStr, log_path)
