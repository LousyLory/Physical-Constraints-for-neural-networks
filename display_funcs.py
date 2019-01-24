import matplotlib.pyplot as plt
from src.vis_utils import visualize_grid
import numpy as np

def save_net_weights(net, name):
    ext = '.pdf'
    W1 = net.params['W1']
    W1 = W1.reshape(3, 32, 32, -1).transpose(3, 1, 2, 0)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.savefig('all_outputs/'+name+'w1'+ext, dpi=300)
    plt.savefig('all_outputs/'+name+'w1'+'.png', dpi=300)
    plt.clf()
    #plt.savefig('')

    W2 = net.params['W2']
    W2 = W2.reshape(1, 10, 10, -1).transpose(3, 1, 2, 0)
    plt.imshow(np.squeeze(visualize_grid(W2, padding=3)).astype('uint8'))
    plt.gca().axis('off')
    plt.savefig('all_outputs/'+name+'w2'+ext, dpi=300)
    plt.savefig('all_outputs/'+name+'w2'+'.png', dpi=300)
    plt.clf()
    return None

def plot_train_loss(solver1, plot_name):
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(solver1.loss_history, 'o')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(solver1.train_acc_history, '-o', label='train')
    plt.plot(solver1.val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(solver1.val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.savefig('all_outputs/'+plot_name+'.pdf')
    plt.savefig('all_outputs/'+plot_name+'.png')
    plt.clf()
    return None
