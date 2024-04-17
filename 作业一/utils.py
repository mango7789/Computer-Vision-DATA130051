from __init__ import *

def download_minist() -> Dict:
    """
    Download the minist dataset from the website and save them in the local directory `data` .

    Return: 
    - data: a dictionary containing the keys ["X_train", "y_train", "X_val", "y_val"] and corresponding
      images and labels.
    """
    urls = {
        "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
    }

    download = not os.path.exists('data')

    if download:
        os.makedirs('data')

    train_images = download_and_extract_data(urls["train_images"], os.path.join("data", "train-images-idx3-ubyte.gz"), download)
    train_labels = download_and_extract_data(urls["train_labels"], os.path.join("data", "train-labels-idx1-ubyte.gz"), download)
    test_images = download_and_extract_data(urls["test_images"], os.path.join("data", "t10k-images-idx3-ubyte.gz"), download)
    test_labels = download_and_extract_data(urls["test_labels"], os.path.join("data", "t10k-labels-idx1-ubyte.gz"), download)

    data = {
        'X_train': train_images,
        'y_train': train_labels,
        'X_val': test_images,
        'y_val': test_labels
    }

    return data

def download_and_extract_data(url: str, file_name: str, download=True):
    """
    Download the data from the given url, load it as a numpy array(matrix) and reshape it as size 28*28.

    Input:
    - url: the url of the dataset
    - file_name: the storage path of the dataset
    - download: whether to download the dataset, defaule is True
    """
    if download:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=file_name) as progress_bar:
            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    progress_bar.total = total_size
                    progress_bar.update(block_size)
            
            urllib.request.urlretrieve(url, file_name, reporthook=reporthook)
    
    with gzip.open(file_name, 'rb') as f:
        if 'images' in file_name:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            data = data.reshape((-1, 28, 28))
        else:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data
    
def plot_stats(stat_dict):
    # plot the loss function and train / validation accuracies
    plt.subplot(1, 2, 1)
    plt.plot(stat_dict['loss_history'], 'o')
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(stat_dict['train_acc_history'], 'o-', label='train')
    plt.plot(stat_dict['val_acc_history'], 'o-', label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()

    plt.gcf().set_size_inches(14, 4)
    plt.show()


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D array of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    # print(Xs.shape)
    grid_size = int(math.ceil(math.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


def show_net_weights(net):
    """
    Visualize the weights of the network
    """
    W1 = net.params['W1']
    W1 = np.transpose(W1.reshape(3, 28, 28, -1), (3, 1, 2, 0))
    plt.imshow(visualize_grid(W1, padding=3).astype(np.uint8))
    plt.gca().axis('off')
    plt.show()


def plot_acc_curves(stat_dict):
    plt.subplot(1, 2, 1)
    for key, single_stats in stat_dict.items():
        plt.plot(single_stats['train_acc_history'], label=str(key))
    plt.title('Train accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')

    plt.subplot(1, 2, 2)
    for key, single_stats in stat_dict.items():
        plt.plot(single_stats['val_acc_history'], label=str(key))
    plt.title('Validation accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()

    plt.gcf().set_size_inches(14, 5)
    plt.show()

