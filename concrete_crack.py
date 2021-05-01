import argparse
import json
import math
import os

import matplotlib
import numpy as np
from PIL import Image
import torch
import torch.distributions as dist
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from trixi.logger import PytorchVisdomLogger
from trixi.logger.experiment.pytorchexperimentlogger import PytorchExperimentLogger
from trixi.util import Config
from trixi.util.pytorchutils import set_seed

from models.enc_dec import Encoder, Generator

# matplotlib.use("Agg") #  , warn=True)


parser = argparse.ArgumentParser(description="Train VAE")
parser.add_argument("--gpu", type=int, help="Index of a free gpu in range [0, 3], if not specified CPU will be used")
parser.add_argument("--visualize", type=bool, default=False)


class VAE(nn.Module):
    def __init__(self, z=20, input_size=784):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_size, 400)
        self.fc21 = nn.Linear(400, z)
        self.fc22 = nn.Linear(400, z)
        self.fc3 = nn.Linear(z, 400)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logstd = self.encode(x)
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd


class VAEConv(torch.nn.Module):
    def __init__(self, input_size, h_size, z_dim, to_1x1=True, conv_op=torch.nn.Conv2d,
                 upsample_op=torch.nn.ConvTranspose2d, normalization_op=None, activation_op=torch.nn.LeakyReLU,
                 conv_params=None, activation_params=None, block_op=None, block_params=None, output_channels=None,
                 additional_input_slices=None, output_activation=None,
                 *args, **kwargs):

        super().__init__()

        input_size_enc = list(input_size)
        input_size_dec = list(input_size)
        if output_channels is not None:
            input_size_dec[0] = output_channels
        if additional_input_slices is not None:
            input_size_enc[0] += additional_input_slices * 2

        self.encoder = Encoder(image_size=input_size_enc, h_size=h_size, z_dim=z_dim * 2,
                               normalization_op=normalization_op, to_1x1=to_1x1, conv_op=conv_op,
                               conv_params=conv_params,
                               activation_op=activation_op, activation_params=activation_params, block_op=block_op,
                               block_params=block_params)
        self.decoder = Generator(image_size=input_size_dec, h_size=h_size[::-1], z_dim=z_dim,
                                 normalization_op=normalization_op, to_1x1=to_1x1, upsample_op=upsample_op,
                                 conv_params=conv_params, activation_op=activation_op,
                                 activation_params=activation_params, block_op=block_op,
                                 block_params=block_params, output_activation=output_activation)

        self.hidden_size = self.encoder.output_size

    def forward(self, inpt, sample=None, **kwargs):
        enc = self.encoder(inpt, **kwargs)

        mu, log_std = torch.chunk(enc, 2, dim=1)
        std = torch.exp(log_std)
        z_dist = dist.Normal(mu, std)

        if sample or self.training:
            z = z_dist.rsample()
        else:
            z = mu

        x_rec = self.decoder(z, **kwargs)

        return x_rec, mu, std

    def encode(self, inpt, **kwargs):
        enc = self.encoder(inpt, **kwargs)
        mu, log_std = torch.chunk(enc, 2, dim=1)
        return mu, log_std

    def decode(self, inpt, **kwargs):
        x_rec = self.decoder(inpt, **kwargs)
        return x_rec


class ConcreteCracksDataset(torch.utils.data.Dataset):

    """
    Concrete Crack Images for Classification Dataset.

    http://dx.doi.org/10.17632/5y9wdsg2zt.2
    """

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        self.train_split = 0.8
        self.test_split = 0.2

        self.data_dir = os.path.join(self.root_dir, "Negative")
        self.train_index = round(len(os.listdir(self.data_dir)) * self.train_split)

        self.train_length = len(os.listdir(self.data_dir)[:self.train_index])
        self.test_length = len(os.listdir(self.data_dir)[self.train_index:])

    def __len__(self):
        if self.train:
            return self.train_length
        else:
            return self.test_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            image_path = os.path.join(self.data_dir, os.listdir(self.data_dir)[idx])
        else:
            image_path = os.path.join(self.data_dir, os.listdir(self.data_dir)[self.train_index + idx])

        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)

        return img


def loss_function(recon_x, x, mu, logstd, rec_log_std=0):
    rec_std = math.exp(rec_log_std)
    rec_var = rec_std ** 2

    x_dist = dist.Normal(recon_x, rec_std)
    log_p_x_z = torch.sum(x_dist.log_prob(x), dim=1)

    z_prior = dist.Normal(0, 1.)
    z_post = dist.Normal(mu, torch.exp(logstd))
    kl_div = torch.sum(dist.kl_divergence(z_post, z_prior), dim=1)

    return torch.mean(kl_div - log_p_x_z), kl_div, -log_p_x_z


def train(epoch, model, optimizer, train_loader, device, scaling, vlog, elog, log_var_std):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        data_flat = data
        # data_flat = data.flatten(start_dim=1).repeat(1, scaling)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data_flat)
        loss, kl, rec = loss_function(recon_batch, data_flat, mu, logvar, log_var_std)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
            # vlog.show_value(torch.mean(kl).item(), name="Kl-loss", tag="Losses")
            # vlog.show_value(torch.mean(rec).item(), name="Rec-loss", tag="Losses")
            # vlog.show_value(loss.item(), name="Total-loss", tag="Losses")

            elog.show_value(torch.mean(kl).item(), name="Kl-loss", tag="Losses")
            elog.show_value(torch.mean(rec).item(), name="Rec-loss", tag="Losses")
            elog.show_value(loss.item(), name="Total-loss", tag="Losses")
            elog.show_value(loss.item() / len(data), name="Train-loss")
            # elog.show_value(np.mean(rec_loss), name="Norm-Rec-loss", tag="Anno")
            # elog.show_value(np.mean(test_loss), name="Norm-Total-loss", tag="Anno")

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(model, test_loader, device, scaling, vlog, elog, image_size, batch_size, log_var_std, size_of_image_side):
    model.eval()
    test_loss = []
    kl_loss = []
    rec_loss = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            data_flat = data
            # data_flat = data.flatten(start_dim=1).repeat(1, scaling)
            recon_batch, mu, logvar = model(data_flat)
            loss, kl, rec = loss_function(recon_batch, data_flat, mu, logvar, log_var_std)
            test_loss += (kl + rec).tolist()
            kl_loss += kl.tolist()
            rec_loss += rec.tolist()
            # if i == 0:
            # n = min(data.size(0), 8)
            # comparison = torch.cat([data[:n],
            #                         recon_batch[:, :image_size].view(batch_size, 3, size_of_image_side, size_of_image_side)[:n]])
            # vlog.show_image_grid(comparison.cpu(),   name='reconstruction')

    # vlog.show_value(np.mean(kl_loss), name="Norm-Kl-loss", tag="Anno")
    # vlog.show_value(np.mean(rec_loss), name="Norm-Rec-loss", tag="Anno")
    # vlog.show_value(np.mean(test_loss), name="Norm-Total-loss", tag="Anno")
    # elog.show_value(np.mean(kl_loss), name="Norm-Kl-loss", tag="Anno")
    # elog.show_value(np.mean(rec_loss), name="Norm-Rec-loss", tag="Anno")
    # elog.show_value(np.mean(test_loss), name="Norm-Total-loss", tag="Anno")
    #
    # test_loss_ab = []
    # kl_loss_ab = []
    # rec_loss_ab = []
    # with torch.no_grad():
    #     for i, (data, _) in enumerate(test_loader_abnorm):
    #         data = data.to(device)
    #         data_flat = data.flatten(start_dim=1).repeat(1, scaling)
    #         recon_batch, mu, logvar = model(data_flat)
    #         loss, kl, rec = loss_function(recon_batch, data_flat, mu, logvar, log_var_std)
    #         test_loss_ab += (kl + rec).tolist()
    #         kl_loss_ab += kl.tolist()
    #         rec_loss_ab += rec.tolist()
    #         if i == 0:
    #             n = min(data.size(0), 8)
    #             comparison = torch.cat([data[:n],
    #                                     recon_batch[:, :image_size].view(batch_size, 1, 28, 28)[:n]])
    #             # vlog.show_image_grid(comparison.cpu(),                                     name='reconstruction2')

    print('====> Test set loss: {:.4f}'.format(np.mean(test_loss)))

    # vlog.show_value(np.mean(kl_loss_ab), name="Unorm-Kl-loss", tag="Anno")
    # vlog.show_value(np.mean(rec_loss_ab), name="Unorm-Rec-loss", tag="Anno")
    # vlog.show_value(np.mean(test_loss_ab), name="Unorm-Total-loss", tag="Anno")
    # elog.show_value(np.mean(kl_loss_ab), name="Unorm-Kl-loss", tag="Anno")
    # elog.show_value(np.mean(rec_loss_ab), name="Unorm-Rec-loss", tag="Anno")
    # elog.show_value(np.mean(test_loss_ab), name="Unorm-Total-loss", tag="Anno")

    # kl_roc, kl_pr = elog.get_classification_metrics(kl_loss + kl_loss_ab,
    #                                                 [0] * len(kl_loss) + [1] * len(kl_loss_ab),
    #                                                 )[0]
    # rec_roc, rec_pr = elog.get_classification_metrics(rec_loss + rec_loss_ab,
    #                                                   [0] * len(rec_loss) + [1] * len(rec_loss_ab),
    #                                                   )[0]
    # loss_roc, loss_pr = elog.get_classification_metrics(test_loss + test_loss_ab,
    #                                                     [0] * len(test_loss) + [1] * len(test_loss_ab),
    #                                                     )[0]

    # vlog.show_value(np.mean(kl_roc), name="KL-loss", tag="ROC")
    # vlog.show_value(np.mean(rec_roc), name="Rec-loss", tag="ROC")
    # vlog.show_value(np.mean(loss_roc), name="Total-loss", tag="ROC")
    # elog.show_value(np.mean(kl_roc), name="KL-loss", tag="ROC")
    # elog.show_value(np.mean(rec_roc), name="Rec-loss", tag="ROC")
    # elog.show_value(np.mean(loss_roc), name="Total-loss", tag="ROC")

    # vlog.show_value(np.mean(kl_pr), name="KL-loss", tag="PR")
    # vlog.show_value(np.mean(rec_pr), name="Rec-loss", tag="PR")
    # vlog.show_value(np.mean(loss_pr), name="Total-loss", tag="PR")

    # return kl_roc, rec_roc, loss_roc, kl_pr, rec_pr, loss_pr
    return np.mean(test_loss), np.mean(kl_loss), np.mean(rec_loss)


def model_run(scaling, batch_size, odd_class, z, resize_data: bool, transform_image_range:bool, gpu_id: int, seed=123, log_var_std=0.0, n_epochs=25):
    set_seed(seed)

    transformation_list = []

    if resize_data:
        size_of_image_side = 64
        transformation_list.append(transforms.Resize((size_of_image_side, size_of_image_side)))
    else:
        size_of_image_side = 64
        transformation_list.append(transforms.CenterCrop((size_of_image_side, size_of_image_side)))

    transformation_list.append(transforms.ToTensor())

    if transform_image_range:
        transformation_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    transform_functions = transforms.Compose(transformation_list)

    config = Config(
        scaling=scaling, batch_size=batch_size, odd_class=odd_class, z=z, resize_data=resize_data,
        transform_image_range=transform_image_range, seed=seed, log_var_std=log_var_std, n_epochs=n_epochs,
        size_of_image_side=size_of_image_side
    )

    if gpu_id is not None:
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        # Use cpu
        device = torch.device("cpu")

    image_size = size_of_image_side * size_of_image_side * 3
    input_shape = (batch_size, 3, size_of_image_side, size_of_image_side)

    model_h_size = (16, 32, 64, 256)

    output_activation = None

    if transform_image_range:
        output_activation = torch.nn.Tanh

    model = VAEConv(input_size=input_shape[1:], h_size=model_h_size, z_dim=z, output_activation=output_activation).to(device)
    # model = VAE(z=z, input_size=input_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=1)

    kwargs = {'num_workers': 0, 'pin_memory': True}

    # train_set = ConcreteCracksDataset(root_dir="/home/pdeubel/PycharmProjects/data/Concrete-Crack-Images", train=True,
    #                                   transform=transform_functions)
    #
    # test_set = ConcreteCracksDataset(root_dir="/home/pdeubel/PycharmProjects/data/Concrete-Crack-Images", train=False,
    #                                  transform=transform_functions)

    train_set = ConcreteCracksDataset(root_dir="/cvhci/data/construction/Concrete-Cracks", train=True,
                                      transform=transform_functions)

    test_set = ConcreteCracksDataset(root_dir="/cvhci/data/construction/Concrete-Cracks", train=False,
                                     transform=transform_functions)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

    # test_loader_abnorm = torch.utils.data.DataLoader(test_set, sampler=test_one_set,
    #                                                  batch_size=batch_size, shuffle=False, **kwargs)

    # vlog = PytorchVisdomLogger(exp_name="vae-concrete-cracks")
    vlog = None
    elog = PytorchExperimentLogger(base_dir="logs/concrete-cracks", exp_name="concrete-cracks_vae")

    elog.save_config(config, "config")

    for epoch in range(1, n_epochs + 1):
        train(epoch, model, optimizer, train_loader, device, scaling, vlog, elog, log_var_std)
        elog.save_model(model, "vae_concrete_crack")

    # kl_roc, rec_roc, loss_roc, kl_pr, rec_pr, loss_pr = test(model, test_loader, device,
    #                                                          scaling, vlog, elog,
    #                                                          image_size, batch_size, log_var_std)

    # with open(os.path.join(elog.result_dir, "results.json"), "w") as file_:
    #     json.dump({
    #         "kl_roc": kl_roc, "rec_roc": rec_roc, "loss_roc": loss_roc,
    #         "kl_pr": kl_pr, "rec_pr": rec_pr, "loss_pr": loss_pr,
    #     }, file_, indent=4)

    test_loss, kl_loss, rec_loss = test(model, test_loader, device, scaling, vlog, elog, image_size, batch_size,
                                        log_var_std, size_of_image_side)

    with open(os.path.join(elog.result_dir, "results.json"), "w") as file_:
        json.dump({
            "test_loss": test_loss, "kl_loss": kl_loss, "rec_loss": rec_loss
        }, file_, indent=4)


def view_trained_model():
    # model = VAE(z=100, input_size=227 * 227 * 3)
    # TODO needs to be redone
    input_shape = (batch_size, 3, 64, 64)
    model_h_size = (16, 32, 64, 256)
    model = VAEConv(input_size=input_shape[1:], h_size=model_h_size, z_dim=z).to(torch.device("cpu"))

    model.load_state_dict(torch.load("logs/concrete-cracks/20210428-224848_concrete-cracks_vae/checkpoint/vae_concrete_crack.pth", map_location=torch.device("cpu")))

    for _ in range(5):
        # matplotlib.pyplot.imshow(np.transpose(model.decode(torch.randn(100)).view(3, 224, 224).detach().numpy(), (2, 1, 0)))
        # matplotlib.pyplot.show()

        matplotlib.pyplot.imshow(np.transpose(model.decode(torch.randn(1, 100, 1, 1))[0].detach().numpy(), (2, 1, 0)))
        matplotlib.pyplot.show()


def main(cli_arguments):
    scaling = 1
    batch_size = 128
    odd_class = 0
    z = 256
    seed = 123
    log_var_std = 0.
    n_epochs = 25
    resize_data = True
    transform_image_range = True

    gpu_id = cli_arguments.gpu

    if gpu_id is not None:
        assert 0 <= gpu_id <= 3, "GPU {} invalid, use one between 0 and 3".format(gpu_id)

    if not cli_arguments.visualize:
        model_run(scaling=scaling, batch_size=batch_size, odd_class=odd_class, z=z, resize_data=resize_data,
                  transform_image_range=transform_image_range, gpu_id=gpu_id, seed=seed, log_var_std=log_var_std,
                  n_epochs=n_epochs)
    else:
        view_trained_model()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



