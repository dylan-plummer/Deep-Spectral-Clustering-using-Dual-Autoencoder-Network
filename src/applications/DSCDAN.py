import os
import umap
import time
import metrics
import datetime
import numpy as np
from sklearn import manifold
from sklearn.cluster import KMeans
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.layers import Input
from core.util import print_accuracy,LearningHandler, get_y_preds
from core import Conv

import tensorflow as tf
def run_net(data, params):
    #
    # UNPACK DATA
    #

    x_train_unlabeled, y_train_unlabeled, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']

    print(params['input_shape'])
    inputs_vae = Input(shape=params['input_shape'], name='inputs_vae')
    ConvAE = Conv.ConvAE(inputs_vae,params)
    try:
        ConvAE.vae.load_weights('vae_mnist.h5')
    except OSError:
        print('No pretrained weights available...')

    lh = LearningHandler(lr=params['spec_lr'], drop=params['spec_drop'], lr_tensor=ConvAE.learning_rate,
                         patience=params['spec_patience'])

    lh.on_train_begin()

    n_epochs = 5000
    losses_vae = np.empty((n_epochs,))
    homo_plot = np.empty((n_epochs,))
    nmi_plot = np.empty((n_epochs,))
    ari_plot = np.empty((n_epochs,))

    y_val = np.squeeze(np.asarray(y_val).ravel())  # squeeze into 1D array

    start_time = time.time()
    for i in range(n_epochs):
        # if i==0:
        x_val_y = ConvAE.vae.predict(x_val)[2]
        losses_vae[i] = ConvAE.train_vae(x_val,x_val_y, params['batch_size'])
        #x_val_y = ConvAE.vae.predict(x_val)[2]
        #y_sp = x_val_y.argmax(axis=1)
        #print_accuracy(y_sp, y_val, params['n_clusters'])
        print("Epoch: {}, loss={:2f}".format(i, losses_vae[i]))

        os.makedirs('vae', exist_ok=True)
        os.makedirs('vae_umap', exist_ok=True)

        fig, axs = plt.subplots(2, 4, figsize=(25, 12))
        fig.subplots_adjust(wspace=0.25)

        embedding = ConvAE.encoder.predict(x_val)
        kmeans = KMeans(n_clusters=params['n_clusters'], n_init=30)
        predicted_labels = kmeans.fit_predict(embedding)  # cluster on current embeddings for metric eval
        _, confusion_matrix = get_y_preds(predicted_labels, y_val, params['n_clusters'])

        homo_plot[i] = metrics.acc(y_val, predicted_labels)
        nmi_plot[i] = metrics.nmi(y_val, predicted_labels)
        ari_plot[i] = metrics.ari(y_val, predicted_labels)

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        Z_tsne = tsne.fit_transform(embedding)
        sc = axs[1][0].scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=y_train_unlabeled, cmap=plt.cm.get_cmap("jet", 14))
        axs[1][0].set_title('t-SNE Embeddings')
        axs[1][0].set_xlabel('t-SNE 1')
        axs[1][0].set_ylabel('t-SNE 2')
        axs[1][0].set_xticks([])
        axs[1][0].set_yticks([])
        axs[1][0].spines['right'].set_visible(False)
        axs[1][0].spines['top'].set_visible(False)
        divider = make_axes_locatable(axs[1][0])
        cax = divider.append_axes('right', size='15%', pad=0.05)
        cbar = fig.colorbar(sc, cax=cax, orientation='vertical', ticks=range(params['n_clusters']))
        cbar.ax.set_yticklabels(params['cluster_names'])  # vertically oriented colorbar
        # Create offset transform by 5 points in x direction
        dx = 0 / 72.
        dy = -5 / 72.
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

        # apply offset transform to all cluster ticklabels.
        for label in cbar.ax.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

        reducer = umap.UMAP(transform_seed=36, random_state=36)
        matrix_reduce = reducer.fit_transform(embedding)
        sc = axs[1][1].scatter(matrix_reduce[:, 0], matrix_reduce[:, 1], s=2, c=y_train_unlabeled, cmap=plt.cm.get_cmap("jet", 14))
        axs[1][1].set_title('UMAP Embeddings')
        axs[1][1].set_xlabel('UMAP 1')
        axs[1][1].set_ylabel('UMAP 2')
        axs[1][1].set_xticks([])
        axs[1][1].set_yticks([])
        # Hide the right and top spines
        axs[1][1].spines['right'].set_visible(False)
        axs[1][1].spines['top'].set_visible(False)

        im = axs[1][2].imshow(confusion_matrix, cmap='YlOrRd')
        axs[1][2].set_title('Confusion Matrix')
        axs[1][2].set_xticks(range(params['n_clusters']))
        axs[1][2].set_yticks(range(params['n_clusters']))
        axs[1][2].set_xticklabels(params['cluster_names'], fontsize=8)
        axs[1][2].set_yticklabels(params['cluster_names'], fontsize=8)
        divider = make_axes_locatable(axs[1][2])
        cax = divider.append_axes('right', size='10%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[])

        axs[0][0].plot(losses_vae[:i + 1])
        axs[0][0].set_title('VAE Loss')
        axs[0][0].set_xlabel('epochs')

        axs[0][1].plot(homo_plot[:i + 1])
        axs[0][1].set_title('Homogeneity')
        axs[0][1].set_xlabel('epochs')
        axs[0][1].set_ylim(0, 1)

        axs[0][2].plot(ari_plot[:i + 1])
        axs[0][2].set_title('ARI')
        axs[0][2].set_xlabel('epochs')
        axs[0][2].set_ylim(0, 1)

        axs[0][3].plot(nmi_plot[:i + 1])
        axs[0][3].set_title('NMI')
        axs[0][3].set_xlabel('epochs')
        axs[0][3].set_ylim(0, 1)

        reconstructed_cell = ConvAE.vae.predict(x_val[:0])[0, ..., 0]
        cell_tile = x_val[0, ..., 0]
        cell_tile = cell_tile[:, :128]
        reconstructed_cell_tile = reconstructed_cell[:, :128]
        cell_heatmap = np.vstack((cell_tile, reconstructed_cell_tile))
        axs[1][3].imshow(cell_heatmap, cmap='Reds')
        axs[1][3].set_xticks([])
        axs[1][3].set_yticks([])
        axs[1][3].spines['right'].set_visible(False)
        axs[1][3].spines['top'].set_visible(False)
        axs[1][3].spines['left'].set_visible(False)
        axs[1][3].spines['bottom'].set_visible(False)


        train_time = str(datetime.timedelta(seconds=(int(time.time() - start_time))))
        n_matrices = (i + 1) * params['batch_size'] * 100
        fig.suptitle('Trained on ' + '{:,}'.format(n_matrices) + ' cells\n' + train_time)

        plt.savefig('vae/%d.png' % i)
        plt.close()

        plt.close()

        if i>1:
            if np.abs(losses_vae[i]-losses_vae[i-1])<0.0001:
                print('STOPPING EARLY')
                break

    print("finished training")

    plt.plot(losses_vae)
    plt.title('VAE Loss')
    plt.show()

    x_val_y = ConvAE.vae.predict(x_val)[2]
    # x_val_y = ConvAE.classfier.predict(x_val_lp)
    y_sp = x_val_y.argmax(axis=1)
    print_accuracy(y_sp, y_val, params['n_clusters'])
    from sklearn.metrics import normalized_mutual_info_score as nmi
    y_val = np.squeeze(np.asarray(y_val).ravel())  # squeeze into 1D array
    print(y_sp.shape, y_val.shape)
    nmi_score1 = nmi(y_sp, y_val)
    print('NMI: ' + str(np.round(nmi_score1, 4)))

    embedding = ConvAE.encoder.predict(x_val)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(embedding)
    fig = plt.figure()
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=y_train_unlabeled, cmap=plt.cm.get_cmap("jet", 14))
    plt.colorbar(ticks=range(params['n_clusters']))
    plt.show()


