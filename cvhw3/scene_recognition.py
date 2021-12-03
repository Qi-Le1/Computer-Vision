import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath
from sklearn.metrics import confusion_matrix, accuracy_score

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


def compute_dsift(img, stride, size):
    # To do
    sift = cv2.xfeatures2d.SIFT_create()
    m, n = img.shape[0], img.shape[1]
    key_point = []
    for i in range(0, m, stride):
        for j in range(0, n, stride):
            key_point.append(cv2.KeyPoint(i+size/2, j+size/2, size))

    key, dense_feature = sift.compute(img, key_point)
    return dense_feature


def get_tiny_image(img, output_size):
    # To do
    tiny_image = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
    tiny_image = tiny_image.reshape(-1)

    tiny_image_mean = tiny_image - np.mean(tiny_image)
    tiny_image_product = np.linalg.norm(tiny_image)
    feature = tiny_image_mean/tiny_image_product

    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    # To do
    neighbor = KNeighborsClassifier(n_neighbors=k)
    neighbor.fit(feature_train, label_train)
    label_test_pred = neighbor.predict(feature_test)

    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    output_size = (15,15)
    train_list = []
    train_label_list = []
    for i in range(len(label_train_list)):
        image = cv2.imread(img_train_list[i], 0)
        feature = get_tiny_image(image, output_size)
        train_list.append(feature)
        train_label_list.append(label_classes.index(label_train_list[i]))
    train_label_list = np.array(train_label_list)

    test_list = []
    test_label_list = []
    for i in range(len(label_test_list)):
        image = cv2.imread(img_test_list[i], 0)
        feature = get_tiny_image(image, output_size)
        test_list.append(feature)
        test_label_list.append(label_classes.index(label_test_list[i]))
    test_label_list = np.array(test_label_list)

    test_label_predict = predict_knn(train_list, train_label_list, test_list, 8)
    confusion = confusion_matrix(test_label_list, test_label_predict)
    accuracy = accuracy_score(test_label_list, test_label_predict)
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):
    # To do
    # dic_size = 50
    dense_feature_list_combine = np.vstack(dense_feature_list)
    iteration = 250
    kmeans = KMeans(n_clusters=dic_size, n_init=10, max_iter=iteration)
    kmeans.fit(dense_feature_list_combine)
    vocab = kmeans.cluster_centers_
    return vocab


def compute_bow(feature, vocab):
    # To do
    neighbor = KNeighborsClassifier(n_neighbors=1)
    label = [i for i in range(len(vocab))]
    neighbor.fit(vocab, label)

    prediction = neighbor.predict(feature)
    bow = [0 for _ in range(len(vocab))]
    for i in range(len(prediction)):
        bow[prediction[i]] += 1

    bow_feature = np.array(bow) / np.linalg.norm(np.array(bow))

    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    stride, size = (15, 15)
    train_list = []
    train_label_list = []
    for i in range(len(label_train_list)):
        image = cv2.imread(img_train_list[i], 0)
        feature = compute_dsift(image, stride, size)
        train_list.append(feature)

        train_label_list.append(label_classes.index(label_train_list[i]))
    train_label_list = np.array(train_label_list)

    vocab = build_visual_dictionary(train_list, 50)

    train_bow_list = []
    for i in range(len(label_train_list)):
        calculated_bow = compute_bow(train_list[i], vocab)
        train_bow_list.append(calculated_bow)

    test_label_list = []
    test_bow_list = []
    for i in range(len(label_test_list)):
        image = cv2.imread(img_test_list[i], 0)
        feature = compute_dsift(image, stride, size)

        calculated_bow_test = compute_bow(feature, vocab)
        test_bow_list.append(calculated_bow_test)

        test_label_list.append(label_classes.index(label_test_list[i]))
    test_label_list = np.array(test_label_list)

    test_label_predict = predict_knn(train_bow_list, train_label_list, test_bow_list, 8)
    confusion = confusion_matrix(test_label_list, test_label_predict)
    accuracy = accuracy_score(test_label_list, test_label_predict)
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    # To do
    all_models = []
    for i in range(n_classes):
        all_models.append(SVC(tol=1e-5, probability=True))

    single_label = [[0 for _ in range(len(feature_train))] for _ in range(n_classes)]
    for i in range(len(feature_train)):
        single_label[label_train[i]][i] = 1

    all_models_prediction = []
    for i in range(n_classes):
        all_models[i].fit(feature_train, single_label[i])
        all_models_prediction.append(all_models[i].predict_log_proba(feature_test))

    label_test_pred = []
    curlist = np.zeros(n_classes)
    for i in range(len(feature_test)):
        for j in range(n_classes):
            curlist[j] = all_models_prediction[j][i][1]
        cur_max = curlist.argmax()
        label_test_pred.append(cur_max)

    label_test_pred = np.array(label_test_pred)
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    stride, size = (15, 15)
    train_list = []
    train_label_list = []
    for i in range(len(label_train_list)):
        image = cv2.imread(img_train_list[i], 0)
        feature = compute_dsift(image, stride, size)
        train_list.append(feature)

        train_label_list.append(label_classes.index(label_train_list[i]))
    train_label_list = np.array(train_label_list)

    vocab = build_visual_dictionary(train_list, 50)

    train_bow_list = []
    for i in range(len(label_train_list)):
        calculated_bow = compute_bow(train_list[i], vocab)
        train_bow_list.append(calculated_bow)

    test_label_list = []
    test_bow_list = []
    for i in range(len(label_test_list)):
        image = cv2.imread(img_test_list[i], 0)
        feature = compute_dsift(image, stride, size)

        calculated_bow_test = compute_bow(feature, vocab)
        test_bow_list.append(calculated_bow_test)

        test_label_list.append(label_classes.index(label_test_list[i]))
    test_label_list = np.array(test_label_list)

    test_label_predict = predict_svm(train_bow_list, train_label_list, test_bow_list, len(label_classes))
    confusion = confusion_matrix(test_label_list, test_label_predict)
    accuracy = accuracy_score(test_label_list, test_label_predict)
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


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

if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info(
        "C://Users//Lucky//PycharmProjects//cvhw3")

    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)



