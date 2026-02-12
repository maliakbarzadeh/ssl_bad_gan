
import numpy as np
import torch
from torchvision.datasets import MNIST, SVHN, CIFAR10
from torchvision import transforms
import torchvision.utils as vutils

class DataLoader(object):

    def __init__(self, config, raw_loader, indices, batch_size):
        self.images, self.labels = [], []
        for idx in indices:
            image, label = raw_loader[idx]
            self.images.append(image)
            self.labels.append(label)

        self.images = torch.stack(self.images, 0)
        self.labels = torch.from_numpy(np.array(self.labels, dtype=np.int64)).squeeze()

        if config.dataset == 'mnist':
            self.images = self.images.view(self.images.size(0), -1)

        self.batch_size = batch_size

        self.unlimit_gen = self.generator(True)
        self.len = len(indices)

    def get_zca_cuda(self, reg=1e-6):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        mean = images.mean(0)
        images -= mean.expand_as(images)
        sigma = torch.mm(images.transpose(0, 1), images) / images.size(0)
        U, S, V = torch.svd(sigma)
        components = torch.mm(torch.mm(U, torch.diag(1.0 / torch.sqrt(S) + reg)), U.transpose(0, 1))
        return components, mean

    def apply_zca_cuda(self, components):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        self.images = torch.mm(images, components.transpose(0, 1)).cpu()

    def generator(self, inf=False):
        while True:
            indices = np.arange(self.images.size(0))
            np.random.shuffle(indices)
            indices = torch.from_numpy(indices)
            for start in range(0, indices.size(0), self.batch_size):
                end = min(start + self.batch_size, indices.size(0))
                ret_images, ret_labels = self.images[indices[start: end]], self.labels[indices[start: end]]
                yield ret_images, ret_labels
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return self.len

def get_mnist_loaders(config):
    transform = transforms.Compose([transforms.ToTensor()])
    training_set = MNIST(config.data_root, train=True, download=True, transform=transform)
    dev_set = MNIST(config.data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: config.size_labeled_data // 10]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0])

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set

def get_svhn_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = SVHN(config.data_root, split='train', download=True, transform=transform)
    dev_set = SVHN(config.data_root, split='test', download=True, transform=transform)

    def preprocess(data_set):
        # Normalize labels: some torchvision versions store labels as shape (N,) ints
        # while others store shape (N,1). Handle both cases and map label '10' -> '0'.
        try:
            labels = np.array(data_set.labels)
        except Exception:
            # Fallback: construct from dataset indexing
            labels = np.array([data_set[i][1] for i in range(len(data_set))])

        # Collapse to 1-D
        if labels.ndim > 1:
            labels = labels.reshape(-1)

        labels[labels == 10] = 0

        # Try to write back into dataset.labels in-place if possible
        try:
            data_set.labels[:] = labels
        except Exception:
            try:
                data_set.labels = labels
            except Exception:
                # As last resort, do nothing; dataset indexing will still work
                pass

    preprocess(training_set)
    preprocess(dev_set)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: config.size_labeled_data // 10]] = True
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    labeled_indices, unlabeled_indices = indices[mask], indices
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set

def get_cifar_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = CIFAR10('cifar', train=True, download=True, transform=transform)
    dev_set = CIFAR10('cifar', train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: config.size_labeled_data // 10]] = True
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    labeled_indices, unlabeled_indices = indices[mask], indices
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set


class TwoCirclesDataset:
    def __init__(self, n_samples, noise, factor, shuffle=True):
        from sklearn.datasets import make_circles
        self.X, self.y = make_circles(n_samples=n_samples, noise=noise, factor=factor, shuffle=shuffle, random_state=42)
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.int64)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.y[idx]


def get_two_circles_loaders(config):
    training_set = TwoCirclesDataset(n_samples=config.n_samples, noise=config.noise_level, 
                                     factor=config.factor, shuffle=True)
    dev_set = TwoCirclesDataset(n_samples=config.n_samples // 2, noise=config.noise_level, 
                                factor=config.factor, shuffle=True)
    
    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    
    for i in range(config.num_label):
        mask[np.where(labels == i)[0][: config.size_labeled_data // config.num_label]] = True
    
    labeled_indices, unlabeled_indices = indices[mask], indices
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(config.num_label):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set

