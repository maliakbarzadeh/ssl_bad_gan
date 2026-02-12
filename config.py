
import numpy as np

class mnist_config:
    dataset = 'mnist'
    image_size = 28 * 28
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 3e-3
    enc_lr = 1e-3
    gen_lr = 1e-3

    eval_period = 600
    vis_period = 100

    data_root = 'data'

    size_labeled_data = 100

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200

    seed = 13

    feature_match = True
    top_k = 5
    top1_weight = 1.

    supervised_only = False
    feature_match = True
    p_loss_weight = 1e-4
    p_loss_prob = 0.1
    
    max_epochs = 2000

    pixelcnn_path = 'model/mnist.True.3.best.pixel'

class svhn_config:
    dataset = 'svhn'
    image_size = 3 * 32 * 32
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 1e-3
    enc_lr = 1e-3
    gen_lr = 1e-3
    min_lr = 1e-4

    eval_period = 730
    vis_period = 730

    data_root = 'data'

    size_labeled_data = 1000

    train_batch_size = 64
    train_batch_size_2 = 64
    dev_batch_size = 200

    max_epochs = 900
    ent_weight = 0.1
    pt_weight = 0.8

    p_loss_weight = 1e-4
    p_loss_prob = 0.1

class cifar_config:
    dataset = 'cifar'
    image_size = 3 * 32 * 32
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 6e-4
    enc_lr = 3e-4
    gen_lr = 3e-4

    eval_period = 500
    vis_period = 500

    data_root = 'data'

    size_labeled_data = 4000

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200

    max_epochs = 1200
    vi_weight = 1e-2

class two_circles_config:
    """
    Configuration class for training a GAN on the two circles dataset.
    This configuration implements the "Good SSL that Requires a Bad GAN" paper approach
    for semi-supervised learning on a simple 2D two-circles dataset.
    Attributes:
        dataset (str): Name of the dataset ('two_circles')
        image_size (int): Dimensionality of data points (2D: x, y coordinates)
        num_label (int): Number of real data classes (2: inner and outer circles)
        num_classes (int): Total classes including fake class (3: inner, outer, fake)
        gen_emb_size (int): Generator embedding/latent feature size
        noise_size (int): Dimensionality of input noise to generator
        dis_lr (float): Learning rate for discriminator
        enc_lr (float): Learning rate for encoder
        gen_lr (float): Learning rate for generator
        eval_period (int): Evaluate model every N iterations
        vis_period (int): Visualize results every N iterations
        plot_period (int): Generate detailed plots every N iterations
        data_root (str): Root directory for dataset storage
        size_labeled_data (int): Number of labeled samples per class (10 total)
        train_batch_size (int): Training batch size for main training loop
        train_batch_size_2 (int): Alternative training batch size
        dev_batch_size (int): Batch size for validation/evaluation
        seed (int): Random seed for reproducibility
        feature_match (bool): Enable feature matching loss
        top_k (int): Number of top features to match
        top1_weight (float): Weight for top-1 feature matching
        supervised_only (bool): If True, use only labeled data
        ent_weight (float): Weight for conditional entropy loss on discriminator
        cond_ent_weight (float): Weight for conditional entropy on unlabeled data
        fm_weight (float): Feature matching loss weight
        pt_weight (float): Pull-away term weight (increases generator diversity)
        use_complement_generator (bool): Enable low-density complement sampling
        density_weight (float): Weight for density-based loss
        density_threshold (float): Threshold (log space) for high/low density distinction
        use_vi_entropy (bool): Enable variational inference for entropy maximization
        vi_weight (float): Weight for VI entropy loss (KL divergence)
        max_epochs (int): Maximum number of training epochs (iterations)
        n_samples (int): Number of samples to generate from two circles dataset
        noise_level (float): Gaussian noise standard deviation added to data
        factor (float): Radius ratio of inner circle to outer circle (0.5 = half)
    """
    dataset = 'two_circles'
    image_size = 2  # 2D points (x, y)
    num_label = 2   # Two real classes (inner and outer circle)
    num_classes = 3 # Total classes including fake (inner, outer, fake)

    gen_emb_size = 20
    noise_size = 5

    dis_lr = 3e-3
    enc_lr = 1e-3
    gen_lr = 3e-4          # REDUCED: was 1e-3 → 3e-4 to prevent generator divergence

    eval_period = 200
    vis_period = 100
    plot_period = 200  # Generate detailed plots every N iterations

    data_root = 'data'

    size_labeled_data = 10  # 5 samples per class

    train_batch_size = 50
    train_batch_size_2 = 50
    dev_batch_size = 1000

    seed = 13

    feature_match = True
    top_k = 10
    top1_weight = 1.8

    supervised_only = False
    
    # Bad GAN: Loss weights and flags (based on "Good SSL that Requires a Bad GAN" paper)
    # Discriminator losses:
    ent_weight = 0.2           # Conditional entropy weight (encourages confident predictions)
    cond_ent_weight = 0.2      # Conditional entropy on unlabeled data (NEW! from paper)
    
    # Generator losses:
    feature_match = True         # Feature matching (keep generator near data manifold)
    fm_weight = 0.8              # Feature matching weight
    pt_weight = 0.2              # Pull-away term weight (increase diversity/entropy)
    
    # Optional: complement generator techniques
    use_complement_generator = True  # Use low-density complement sampling (requires density model)
    density_weight = 0.5       # Weight for density-based loss (if density model available)
    density_threshold = 0.05   # Threshold ε for distinguishing high/low density (log space)
    
    # Optional: Variational Inference for entropy maximization
    use_vi_entropy = True     # Use VI-based entropy maximization (requires encoder)
    vi_weight = 0.4            # Weight for VI entropy loss (KL divergence)
    
    max_epochs = 20000
    
    # Two circles specific parameters
    n_samples = 2000
    noise_level = 0.05
    factor = 0.5  # ratio of inner to outer circle radius

class pixelcnn_config:
    dataset = 'mnist'
    image_wh = 28 if dataset == 'mnist' else 32
    n_channel = 1 if dataset == 'mnist' else 3
    image_size = 28 * 28 if dataset == 'mnist' else 32 * 32

    if dataset == 'cifar':
        train_batch_size = 20 * 4
        test_batch_size = 20 * 4
        lr = 1e-3 * 96 / train_batch_size
        disable_third = False
        nr_resnet = 5
        dropout_p = 0.5
    elif dataset == 'svhn':
        train_batch_size = 30 * 4
        test_batch_size = 30 * 4
        lr = 2e-4
        disable_third = True
        nr_resnet = 3
        dropout_p = 0.0
    elif dataset == 'mnist':
        train_batch_size = 40 * 1
        test_batch_size = 40 * 1
        lr = 2e-4
        disable_third = True
        nr_resnet = 3
        dropout_p = 0.0

    eval_period = 30
    save_period = 5
