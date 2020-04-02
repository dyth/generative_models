# Adapted from https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_dataset = datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../data/', train=False, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

# build network
z_dim = 100
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

G = Generator(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)

# loss
criterion = nn.BCELoss()

# optimizer
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)


# def D_train(x):
#     'train the discriminator'
#     D.zero_grad()
#
#     # discriminator loss on real data
#     x_real = x.view(-1, mnist_dim).to(device)
#     output = D(x_real)
#     y_real = torch.ones(batch_size, 1).to(device)
#     D_real_loss = criterion(output, y_real)
#
#     # discriminator loss on fake data
#     # randn outputs values sampled from N(0, 1)
#     z = torch.randn(batch_size, z_dim).to(device)
#     output = D(G(z))
#     y_fake = torch.zeros(batch_size, 1).to(device)
#     D_fake_loss = criterion(output, y_fake)
#
#     # only backprop through discriminator parameters
#     D_loss = D_real_loss + D_fake_loss
#     D_loss.backward()
#     D_optimizer.step()
#     return D_loss.data.item()
#
#
# def G_train(x):
#     'train the generator'
#     G.zero_grad()
#
#     z = torch.randn(batch_size, z_dim).to(device)
#     output = D(G(z))
#     y = torch.ones(batch_size, 1).to(device)
#     G_loss = criterion(output, y)
#
#     # only backprop through generator parameters
#     G_loss.backward()
#     G_optimizer.step()
#     return G_loss.data.item()
#
#
# n_epoch = 100
# for epoch in range(1, n_epoch+1):
#     D_losses, G_losses = [], []
#     for batch_idx, (x, _) in enumerate(train_loader):
#         D_losses.append(D_train(x))
#         G_losses.append(G_train(x))
#     print('[%d/%d]: D loss: %.3f, G loss: %.3f' % (
#         epoch, n_epoch,
#         torch.mean(torch.FloatTensor(D_losses)),
#         torch.mean(torch.FloatTensor(G_losses)))
#     )



# def train(x):
#     'train the generator and discriminator for batch x'
#     # generate data and labels for a batch
#     fake_labels = torch.zeros(batch_size, 1).to(device)
#     real_labels = torch.ones(batch_size, 1).to(device)
#     G_optimizer.zero_grad()
#     z = torch.randn(batch_size, z_dim).to(device)
#     fake_data = G(z)
#     real_data = x.view(-1, mnist_dim).to(device)
#
#     # train the generator parameters only
#     G_loss = criterion(D(fake_data), real_labels)
#     G_loss.backward()
#     G_optimizer.step()
#
#     # train the discriminator parameters only
#     D_optimizer.zero_grad()
#     D_real_loss = criterion(D(real_data), real_labels)
#     D_fake_loss = criterion(D(fake_data.detach()), fake_labels)
#     D_loss = D_real_loss + D_fake_loss
#     D_loss.backward()
#     D_optimizer.step()
#     return G_loss.data.item(), D_loss.data.item()



def train(x):
    y_real = torch.ones(batch_size, 1).to(device)
    y_fake = torch.zeros(batch_size, 1).to(device)
    x_real = x.view(-1, mnist_dim).to(device)

    # train the discriminator
    D.zero_grad()
    # randn outputs values sampled from N(0, 1)
    z = torch.randn(batch_size, z_dim).to(device)
    D_fake_loss = criterion(D(G(z)), y_fake)
    D_real_loss = criterion(D(x_real), y_real)
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    # train the generator
    G.zero_grad()
    z = torch.randn(batch_size, z_dim).to(device)
    G_loss = criterion(D(G(z)), y_real)
    G_loss.backward()
    G_optimizer.step()
    return G_loss.data.item(), D_loss.data.item()




n_epoch = 10
for epoch in range(1, n_epoch+1):
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        G_loss, D_loss = train(x)
        D_losses.append(D_loss)
        G_losses.append(G_loss)
    print('[%d/%d]: D loss: %.3f, G loss: %.3f' % (
        epoch, n_epoch,
        torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses)))
    )


with torch.no_grad():
    test_z = torch.randn(batch_size, z_dim).to(device)
    generated = G(test_z)
    save_image(generated.view(generated.size(0), 1, 28, 28), 'sample.png')
