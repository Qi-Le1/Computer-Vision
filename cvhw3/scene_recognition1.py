import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(
            PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


def compute_dsift(img, stride, size):
    assert size <= img.shape[0] and size <= img.shape[1]
    r = int((img.shape[0] - size) / stride)
    c = int((img.shape[1] - size) / stride)
    sift = cv2.xfeatures2d.SIFT_create()
    dense_feature = []

    for i in range(r):
        for j in range(c):
            patch = img[i * stride: i * stride + size, j * stride: j * stride + size]
            kps, patch_descriptors = sift.compute(
                img,
                [cv2.KeyPoint(x=j * stride + size, y=i * stride + size, _size=size)]
            )
            for descriptor in patch_descriptors:
                dense_feature.append(descriptor.reshape(-1))
    dense_feature = np.array(dense_feature)
    return dense_feature


def build_visual_dictionary(dense_feature_list, dic_size):
    dense_feature_vstack = np.vstack(dense_feature_list)
    INIT = 'k-means++'
    N_INIT = 10
    MAX_ITER = 300
    RANDOM_STATE = 0
    print(
        'K-means clustering started with params (init, n_init, max_iter, random_state) = ({}, {}, {}, {}) on data of shape {}'.format(
            INIT, N_INIT, MAX_ITER, RANDOM_STATE, dense_feature_vstack.shape))
    kmeans = KMeans(dic_size, init=INIT, n_init=N_INIT, max_iter=MAX_ITER, random_state=RANDOM_STATE)
    kmeans.fit(dense_feature_vstack)
    return kmeans.cluster_centers_


def compute_bow(feature, vocab):
    dic_size = vocab.shape[0]
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(vocab, [i for i in range(dic_size)])
    classes = neigh.predict(feature)
    bow = [0 for i in range(dic_size)]
    for _class in classes:
        bow[_class] += 1
    bow = np.asarray(bow)
    bow = bow / np.linalg.norm(bow)
    return bow


def predict_knn(feature_train, label_train, feature_test, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(feature_train, label_train)
    test_labels = neigh.predict(feature_test)
    return test_labels


def classify_knn_bow(class_labels, train_labels, train_images, true_test_labels, test_images):
    STRIDE = 32
    SHAPE = 32
    DIC_SIZE = 50
    print('In dense sift calculation, using stride: {} shape: {}'.format(STRIDE, SHAPE))
    KNN_K = 8
    # Calculating train dense features
    train_dense_feature_list = []
    for i, train_image_file in enumerate(train_images):
        print('Computing dense sift for train images {}/{}\r'.format(i + 1, len(train_images)), end='')
        train_image = cv2.imread(train_image_file, 0)
        dense_feature = compute_dsift(train_image, STRIDE, SHAPE)
        train_dense_feature_list.append(dense_feature)
    print()

    # Calculating vocabulary
    vocab = build_visual_dictionary(train_dense_feature_list, DIC_SIZE)
    np.savetxt('vocab.txt', vocab)
    # vocab = np.loadtxt('vocab.txt')
    print('Vocab created with shape {}'.format(vocab.shape))

    # Calculating train bow features
    train_bows = []
    for i, dense_feature in enumerate(train_dense_feature_list):
        print('Computing BOW for train images {}/{}\r'.format(i + 1, len(train_images)), end='')
        train_bows.append(compute_bow(dense_feature, vocab))
    print()
    train_bows = np.asarray(train_bows)

    # Calculating test bow features
    test_bows = []
    for i, test_image_file in enumerate(test_images):
        print('Computing BOW for test images {}/{}\r'.format(i + 1, len(test_images)), end='')
        test_image = cv2.imread(test_image_file, 0)
        dense_feature = compute_dsift(test_image, STRIDE, SHAPE)
        test_bows.append(compute_bow(dense_feature, vocab))
    print()
    test_bows = np.asarray(test_bows)

    train_labels_encoded = []
    for train_label in train_labels:
        train_labels_encoded.append(class_labels.index(train_label))
    predicted_test_labels_encoded = predict_knn(train_bows, train_labels_encoded, test_bows, KNN_K)
    predicted_test_labels = [class_labels[encoded_label] for encoded_label in predicted_test_labels_encoded]
    confusion = confusion_matrix(true_test_labels, predicted_test_labels)
    accuracy = accuracy_score(true_test_labels, predicted_test_labels)
    print('BOW+KNN : kNN\'s k: {} accuracy: {}'.format(KNN_K, accuracy))

    visualize_confusion_matrix(confusion, accuracy, class_labels)
    return confusion, accuracy


def get_tiny_image(img, output_size):
    h, w = output_size
    h_stride = int(img.shape[0] / h)
    w_stride = int(img.shape[1] / w)
    feature = np.zeros(output_size)
    for hi in range(h):
        for wi in range(w):
            feature[hi, wi] = np.average(img[hi * h_stride: (hi + 1) * h_stride, wi * w_stride: (wi + 1) * w_stride])
    feature = feature.reshape(-1)
    norm = np.linalg.norm(feature)
    feature = feature / norm
    mean = np.mean(feature)
    feature = feature - mean
    return feature


def classify_knn_tiny(class_labels, train_labels, train_images, true_test_labels, test_images):
    TINY_FEATURE_SIZE = (16, 16)
    K_IN_KNN = 6
    # Create train features
    feature_train = []
    for img_file in train_images:
        img = cv2.imread(img_file, 0)
        tiny_img = get_tiny_image(img, TINY_FEATURE_SIZE)
        feature_train.append(tiny_img)
    feature_train = np.asarray(feature_train)

    # Create test features
    feature_test = []
    for img_file in test_images:
        img = cv2.imread(img_file, 0)
        tiny_img = get_tiny_image(img, TINY_FEATURE_SIZE)
        feature_test.append(tiny_img)
    feature_test = np.asarray(feature_test)

    train_labels_encoded = []
    for train_label in train_labels:
        train_labels_encoded.append(class_labels.index(train_label) + 1)
    predicted_test_labels_encoded = predict_knn(feature_train, train_labels_encoded, feature_test, K_IN_KNN)
    predicted_test_labels = [class_labels[encoded_label - 1] for encoded_label in predicted_test_labels_encoded]
    confusion = confusion_matrix(true_test_labels, predicted_test_labels)
    accuracy = accuracy_score(true_test_labels, predicted_test_labels)
    print('TINY+KNN : tiny_size: {} kNN\'s k: {} accuracy: {}'.format(TINY_FEATURE_SIZE, K_IN_KNN, accuracy))

    visualize_confusion_matrix(confusion, accuracy, class_labels)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    C = 4
    GAMMA = 'scale'
    print('Using C = {} GAMMA = {} for SVC'.format(C, GAMMA))
    models = []
    for i in range(n_classes):
        models.append(SVC(random_state=0, tol=1e-5, C=C, gamma=GAMMA, probability=True))
    for i, model in enumerate(models):
        binary_labels = []
        for label in label_train:
            if i + 1 == label:
                binary_labels.append(1)
            else:
                binary_labels.append(0)
        model.fit(feature_train, binary_labels)
    # predict all feature test samples for all models
    across_model_predictions = []
    for i, model in enumerate(models):
        across_model_predictions.append(model.predict_log_proba(feature_test))
    across_model_predictions = np.asarray(across_model_predictions)

    # choose the class of corresponding model that gives highest probability
    predicted_test_labels = []
    for i, test_sample in enumerate(feature_test):
        predicted_class = across_model_predictions[:, i, 1].argmax() + 1
        # print(across_model_predictions[:, i, 1])
        # print(predicted_class)
        predicted_test_labels.append(predicted_class)

    predicted_test_labels = np.asarray(predicted_test_labels)
    return predicted_test_labels


def classify_svm_bow_load(class_labels, train_labels, train_images, true_test_labels, test_images):
    vocab = np.loadtxt('vocab.txt')
    print('Vocab created with shape {}'.format(vocab.shape))
    train_bows = np.loadtxt('train_bows.txt')
    test_bows = np.loadtxt('test_bows.txt')

    train_labels_encoded = []
    for train_label in train_labels:
        train_labels_encoded.append(class_labels.index(train_label) + 1)
    predicted_test_labels_encoded = predict_svm(train_bows, train_labels_encoded, test_bows, len(class_labels))
    predicted_test_labels = [class_labels[encoded_label - 1] for encoded_label in predicted_test_labels_encoded]
    confusion = confusion_matrix(true_test_labels, predicted_test_labels)
    accuracy = accuracy_score(true_test_labels, predicted_test_labels)
    print('BOW+SVC : accuracy: {}'.format(accuracy))

    visualize_confusion_matrix(confusion, accuracy, class_labels)
    return confusion, accuracy


def classify_svm_bow(class_labels, train_labels, train_images, true_test_labels, test_images):
    STRIDE = 32
    SHAPE = 24
    DIC_SIZE = 64
    print('In dense sift calculation, using stride: {} shape: {}'.format(STRIDE, SHAPE))
    # Calculating train dense features
    train_dense_feature_list = []
    for i, train_image_file in enumerate(train_images):
        print('Computing dense sift for train images {}/{}\r'.format(i + 1, len(train_images)), end='')
        train_image = cv2.imread(train_image_file, 0)
        dense_feature = compute_dsift(train_image, STRIDE, SHAPE)
        train_dense_feature_list.append(dense_feature)
    print()

    # test_dense_feature_list = []
    # for i, test_image_file in enumerate(test_images):
    #     print('Computing dense sift for test images {}/{}\r'.format(i + 1, len(test_images)), end='')
    #     test_image = cv2.imread(test_image_file, 0)
    #     dense_feature = compute_dsift(test_image, STRIDE, SHAPE)
    #     test_dense_feature_list.append(dense_feature)
    # print()

    # Calculating vocabulary
    vocab = build_visual_dictionary(train_dense_feature_list, DIC_SIZE)
    np.savetxt('vocab.txt', vocab)
    # vocab = np.loadtxt('vocab.txt')
    print('Vocab created with shape {}'.format(vocab.shape))

    # Calculating train bow features
    train_bows = []
    for i, dense_feature in enumerate(train_dense_feature_list):
        print('Computing BOW for train images {}/{}\r'.format(i + 1, len(train_images)), end='')
        train_bows.append(compute_bow(dense_feature, vocab))
    print()
    train_bows = np.asarray(train_bows)
    np.savetxt('train_bows.txt', train_bows)
    # train_bows = np.loadtxt('train_bows.txt')

    # Calculating test bow features
    test_bows = []

    # for i, dense_feature in enumerate(test_dense_feature_list):
    #     print('Computing BOW for test images {}/{}\r'.format(i + 1, len(test_images)), end='')
    #     test_bows.append(compute_bow(dense_feature, vocab))
    # print()

    for i, test_image_file in enumerate(test_images):
        print('Computing BOW for test images {}/{}\r'.format(i + 1, len(test_images)), end='')
        test_image = cv2.imread(test_image_file, 0)
        dense_feature = compute_dsift(test_image, STRIDE, SHAPE)
        test_bows.append(compute_bow(dense_feature, vocab))
    print()

    test_bows = np.asarray(test_bows)
    np.savetxt('test_bows.txt', test_bows)
    # test_bows = np.loadtxt('test_bows.txt')

    train_labels_encoded = []
    for train_label in train_labels:
        train_labels_encoded.append(class_labels.index(train_label) + 1)
    predicted_test_labels_encoded = predict_svm(train_bows, train_labels_encoded, test_bows, len(class_labels))
    predicted_test_labels = [class_labels[encoded_label - 1] for encoded_label in predicted_test_labels_encoded]
    confusion = confusion_matrix(true_test_labels, predicted_test_labels)
    accuracy = accuracy_score(true_test_labels, predicted_test_labels)
    print('BOW+SVC : accuracy: {}'.format(accuracy))

    visualize_confusion_matrix(confusion, accuracy, class_labels)
    return confusion, accuracy


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list \
        = extract_dataset_info("C://Users//Lucky//PycharmProjects//cvhw3")

    # classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    # classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)




