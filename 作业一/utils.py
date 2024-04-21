from __init__ import *
from solve import Solver

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
    
def plot_stats_single(net: Solver, window_size: int=100):
    """
    Plot the loss function and train / validation accuracies for one single net.
    """
    plt.subplot(2, 1, 1)
    # plt.plot(net.loss_hist, 'o', label='discrete loss')
    plt.plot([sum(net.loss_hist[i:i+window_size])/window_size for i in range(len(net.loss_hist)-window_size)], 'red', label='moving average')
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(net.train_acc_hist, 'o-', label='train')
    plt.plot(net.val_acc_hist, 'o-', label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()

    plt.gcf().set_size_inches(14, 10)
    plt.show()

def plot_acc_multi(nets: List[Solver]):
    """
    Plot the training and validation accuracies for multiple nets in one image.
    """
    plt.subplot(1, 2, 1)
    for net in nets:
        plt.plot(net.train_acc_hist, label=str(net))
    plt.title('Train accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')

    plt.subplot(1, 2, 2)
    for net in nets:
        plt.plot(net.val_acc_hist, label=str(net))
    plt.title('Validation accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()

    plt.gcf().set_size_inches(14, 5)
    plt.show()

def show_net_weights(net: Solver):
    """
    Visualize the weights of the network
    """
    weights = [val for key, val in net.model.params.items() if key[0] == 'W']
    max_shape = np.max([weight.shape for weight in weights], axis=0)
    fig, axs = plt.subplots(1, len(weights), figsize=(15, 5))

    for index, weight in enumerate(weights):
        img = axs[index].imshow(weight, cmap='viridis', extent=[0, max_shape[1], 0, max_shape[0]])
        axs[index].set_title('W{}'.format(index + 1))
    plt.colorbar(img, ax=axs.ravel().tolist(), shrink=0.65)
    plt.suptitle('Weights of the network', y=0.2, x=0.44)
    plt.show()
