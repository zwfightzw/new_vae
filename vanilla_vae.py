import os
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, kl_divergence

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class Sprites(torch.utils.data.Dataset):
    def __init__(self, path, size):
        self.path = path
        self.length = size;

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.load(self.path + '/%d.sprite' % (idx + 1))


class FullQDisentangledVAE(nn.Module):
    def __init__(self, frames, z_dim, conv_dim, hidden_dim):
        super(FullQDisentangledVAE, self).__init__()
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim

        self.z_lstm = nn.LSTM(self.conv_dim, self.hidden_dim//2, 1,
                              bidirectional=True, batch_first=True)
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_mean_drop = nn.Dropout(0.3)
        self.z_logvar_drop = nn.Dropout(0.3)

        self.z_mean_prior = nn.Linear(self.z_dim, self.z_dim)
        self.z_logvar_prior = nn.Linear(self.z_dim, self.z_dim)

        self.z_to_c_fwd = nn.Linear(self.z_dim, self.z_dim)

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

    def reparameterize(self, mean, logvar):
        if self.training:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def encode_z(self, x):
        lstm_out, _ = self.z_lstm(x)

        post_z_list = []
        prior_z_lost = []
        zt_obs_list = []
        zt_1_mean = self.z_mean(self.z_mean_drop(lstm_out[:,0]))
        zt_1_lar = self.z_logvar(self.z_logvar_drop(lstm_out[:,0]))
        post_z_list.append(Normal(zt_1_mean, zt_1_lar))
        prior_z0 = torch.distributions.Normal(torch.zeros(self.z_dim).to(device),
                                              torch.ones(self.z_dim).to(device))

        prior_z_lost.append(prior_z0)
        # decode z0 observation
        zt_1_dec = self.reparameterize(zt_1_mean, zt_1_lar)

        zt_obs_list.append(zt_1_dec)
        batch_size = lstm_out.shape[0]
        seq_size = lstm_out.shape[1]

        zt_1 = torch.zeros(batch_size, self.z_dim).to(device)
        for t in range(1, seq_size):

            # posterior over ct, q(ct|ot,ft)
            ct_post_mean = self.z_mean(self.z_mean_drop(lstm_out[:, t]))
            ct_post_lar = self.z_logvar(self.z_logvar_drop(lstm_out[:, t]))
            post_z_list.append(Normal(ct_post_mean, ct_post_lar))
            # p(xt|zt)
            zt_obs_list.append(self.reparameterize(ct_post_mean, ct_post_lar))

            # prior over ct of each block, ct_i~p(ct_i|zt-1_i)
            c_fwd = self.z_to_c_fwd(zt_1)
            c_fwd_latent_mean = self.z_mean_prior(c_fwd)
            c_fwd_latent_lar = self.z_logvar_prior(c_fwd)

            # store the prior of ct_i
            prior_z_lost.append(Normal(c_fwd_latent_mean, c_fwd_latent_lar))
            ct = self.reparameterize(c_fwd_latent_mean, c_fwd_latent_lar)
            zt = zt_1 + ct

            zt_1 = zt

        zt_obs_list = torch.stack(zt_obs_list, dim=1)

        return post_z_list, prior_z_lost, zt_obs_list

    def forward(self, x):
        conv_x = self.encode_frames(x)

        post_zt, prior_zt, z = self.encode_z(conv_x)
        recon_x = self.decode_frames(z)
        return post_zt, prior_zt, z, recon_x

def loss_fn(original_seq, recon_seq, post_z, prior_z):
    mse = F.mse_loss(recon_seq, original_seq, reduction='sum');
    # compute kl related to states, kl(q(ct|ot,ft)||p(ct|zt-1)) and kl(q(z0|f0)||N(0,1))
    kl_z_list = []
    for t in range(len(post_z)):
        # kl divergences (sum over dimension)
        kl_obs_state = kl_divergence(post_z[t], prior_z[t])
        kl_z_list.append(kl_obs_state.sum(-1))
    kld_z = torch.stack(kl_z_list, dim=1)

    return mse + kld_z.mean(), mse, kld_z.mean()

class Trainer(object):
    def __init__(self, model, device, train, test, trainloader, testloader, epochs, batch_size, learning_rate, nsamples,
                 sample_path, recon_path, checkpoints):
        self.trainloader = trainloader
        self.train = train
        self.test = test
        self.testloader = testloader
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.samples = nsamples
        self.sample_path = sample_path
        self.recon_path = recon_path
        self.test_z = torch.randn(self.samples, model.frames, model.z_dim, device=self.device)

        self.epoch_losses = []
        '''
        self.image1 = torch.load('image1.sprite')
        self.image2 = torch.load('image2.sprite')
        self.image1 = self.image1.to(device)
        self.image2 = self.image2.to(device)
        self.image1 = torch.unsqueeze(self.image1, 0)
        self.image2 = torch.unsqueeze(self.image2, 0)
        '''

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
            zt_1 = torch.zeros(2, self.model.z_dim).to(device)
            zt_dec.append(zt_1)
            for t in range(1, 8):

                # prior over ct of each block, ct_i~p(ct_i|zt-1_i)
                c_fwd = self.model.z_to_c_fwd(zt_1)
                c_fwd_latent_mean = self.model.z_mean_prior(c_fwd)
                c_fwd_latent_lar = self.model.z_logvar_prior(c_fwd)

                ct = self.model.reparameterize(c_fwd_latent_mean, c_fwd_latent_lar)
                zt = zt_1 + ct

                zt_dec.append(zt)
                zt_1 = zt
            zt_dec = torch.stack(zt_dec, dim=1)
            recon_x = self.model.decode_frames(zt_dec)
            recon_x = recon_x.view(16, 3, 64, 64)
            torchvision.utils.save_image(recon_x, './Vanilla/%s/epoch%d.png' % (self.sample_path, epoch))

    def recon_frame(self, epoch, original):
        with torch.no_grad():
            _, _, _, recon = self.model(original)
            image = torch.cat((original, recon), dim=0)
            print(image.shape)
            image = image.view(16, 3, 64, 64)
            torchvision.utils.save_image(image, './Vanilla/%s/epoch%d.png' % (self.recon_path, epoch))
    '''
    def style_transfer(self, epoch):
        with torch.no_grad():
            conv1 = self.model.encode_frames(self.image1)
            conv2 = self.model.encode_frames(self.image2)
            _, _, image1_f = self.model.encode_f(conv1)
            image1_f_expand = image1_f.unsqueeze(1).expand(-1, self.model.frames, self.model.f_dim)
            _, _, image1_z = self.model.encode_z(conv1)
            _, _, image2_f = self.model.encode_f(conv2)
            image2_f_expand = image2_f.unsqueeze(1).expand(-1, self.model.frames, self.model.f_dim)
            _, _, image2_z = self.model.encode_z(conv2)
            image1swap_zf = torch.cat((image2_z, image1_f_expand), dim=2)
            image1_body_image2_motion = self.model.decode_frames(image1swap_zf)
            image1_body_image2_motion = torch.squeeze(image1_body_image2_motion, 0)
            image2swap_zf = torch.cat((image1_z, image2_f_expand), dim=2)
            image2_body_image1_motion = self.model.decode_frames(image2swap_zf)
            image2_body_image1_motion = torch.squeeze(image2_body_image1_motion, 0)
            os.makedirs(os.path.dirname('./Vanilla/transfer/epoch%d/image1_body_image2_motion.png' % epoch),
                        exist_ok=True)
            torchvision.utils.save_image(image1_body_image2_motion,
                                         './Vanilla/transfer/epoch%d/image1_body_image2_motion.png' % epoch)
            torchvision.utils.save_image(image2_body_image1_motion,
                                         './Vanilla/transfer/epoch%d/image2_body_image1_motion.png' % epoch)
    '''

    def train_model(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            print("Running Epoch : {}".format(epoch + 1))
            for i, data in enumerate(self.trainloader, 1):
                data = data.to(device)
                self.optimizer.zero_grad()
                post_z, prior_z, z, recon_x = self.model(data)
                loss, mse, kl = loss_fn(data, recon_x, post_z, prior_z)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                print('mse loss is %f, kl loss is %f'%(mse.item(), kl.item()))
            meanloss = np.mean(losses)
            self.epoch_losses.append(meanloss)
            print("Epoch {} : Average Loss: {}".format(epoch + 1, meanloss))
            self.save_checkpoint(epoch)
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
    vae = FullQDisentangledVAE(frames=8, z_dim=32, hidden_dim=512, conv_dim=1024)
    sprites_train = Sprites('./dataset/lpc-dataset/train/', 6687)
    sprites_test = Sprites('./dataset/lpc-dataset/test/', 873)
    trainloader = torch.utils.data.DataLoader(sprites_train, batch_size=64, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(sprites_test, batch_size=1, shuffle=True, num_workers=4)

    trainer = Trainer(vae, device, sprites_train, sprites_test, trainloader, testloader, epochs=100, batch_size=64,
                      learning_rate=0.0002, checkpoints='./model/Vanilla-disentangled-vae.model', nsamples=2,
                      sample_path='samples',
                      recon_path='recon')
    trainer.load_checkpoint()
    trainer.train_model()
