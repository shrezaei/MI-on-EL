from __future__ import print_function
import numpy as np
import argparse
import torch
from torch import nn, optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from utils import false_alarm_rate, to_categorical

parser = argparse.ArgumentParser(description='Obtaining outputs from saved models.')
#general arguements
parser.add_argument('--outputs_path', type=str, default='outputs/resnet20/', help='Path to saved outputs')
parser.add_argument('-d', '--dataset', default='cifar10', type=str, choices=['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn'])
parser.add_argument('--output-type', type=str, default='confidence',
                    help='Ensembling based on averaging confidence or logit (default: confidence)')
parser.add_argument('--attack-type', type=str, default='aggregated',
                    help='Ensembling based on averaging aggregated or all (whitebox) (default: aggregated)')

args = parser.parse_args()

#Options based on code of the paper of Rezaei et al "Towards the Difficulty of Membership Inference Attacks"
# sampling = "None"
# sampling = "oversampling"
sampling = "undersampling"
balanceness_ratio = 5
what_portion_of_sampels_attacker_knows = 0.8

def main():
    if args.dataset == "mnist":
        num_classes = 10
    elif args.dataset == "fmnist":
        num_classes = 10
    elif args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "svhn":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    else:
        print("This dataset is not supported in current version!")
        exit()

    labels_train = np.load(args.outputs_path + "/label_train.npy")
    labels_test = np.load(args.outputs_path + "/label_test.npy")

    if len(labels_train.shape) > 1:
        labels_train = labels_train.reshape((-1))
    if len(labels_test.shape) > 1:
        labels_test = labels_test.reshape((-1))

    train_conf_all = np.load(args.outputs_path + "/" + args.output_type + "_train.npy")
    test_conf_all = np.load(args.outputs_path + "/" + args.output_type + "_test.npy")

    number_of_models = int(train_conf_all.shape[-1] / num_classes)

    train_conf_sum = np.zeros((labels_train.shape[0], num_classes))
    test_conf_sum = np.zeros((labels_test.shape[0], num_classes))

    train_prediction_class_sum = np.zeros((labels_train.shape[0], num_classes))
    test_prediction_class_sum = np.zeros((labels_test.shape[0], num_classes))

    for model_index_counter in range(number_of_models):
        train_conf_sum += train_conf_all[:, model_index_counter * num_classes:(model_index_counter + 1) * num_classes]
        test_conf_sum += test_conf_all[:, model_index_counter * num_classes:(model_index_counter + 1) * num_classes]

        temp1 = np.argmax(train_conf_all[:, model_index_counter * num_classes:(model_index_counter + 1) * num_classes], axis=1)
        temp2 = np.argmax(test_conf_all[:, model_index_counter * num_classes:(model_index_counter + 1) * num_classes], axis=1)
        train_prediction_class_sum += to_categorical(temp1, num_classes)
        test_prediction_class_sum += to_categorical(temp2, num_classes)

        if args.output_type == "confidence":
            confidence_train_for_prediction = train_conf_sum / (model_index_counter + 1)
            confidence_test_for_prediction = test_conf_sum / (model_index_counter + 1)
        elif args.output_type == "logit":
            confidence_train_for_prediction = softmax(train_conf_sum / (model_index_counter + 1), axis=1)
            confidence_test_for_prediction = softmax(test_conf_sum / (model_index_counter + 1), axis=1)
        else:
            print("Output type does not exist!")
            exit()

        if args.attack_type == "all":
            confidence_train_for_attack = train_conf_all[:, 0:(model_index_counter + 1) * num_classes]
            confidence_test_for_attack = test_conf_all[:, 0:(model_index_counter + 1) * num_classes]
        elif args.attack_type == "aggregated":
            confidence_train_for_attack = confidence_train_for_prediction
            confidence_test_for_attack = confidence_test_for_prediction
        else:
            print("Attack type is not valid!")
            exit()

        labels_train_by_model = np.argmax(confidence_train_for_prediction, axis=1)
        labels_test_by_model = np.argmax(confidence_test_for_prediction, axis=1)

        acc_train = np.sum(labels_train == labels_train_by_model)/labels_train.shape[0]
        acc_test = np.sum(labels_test == labels_test_by_model)/labels_test.shape[0]

        correctly_classified_indexes_train = labels_train_by_model == labels_train
        incorrectly_classified_indexes_train = labels_train_by_model != labels_train

        correctly_classified_indexes_test = labels_test_by_model == labels_test
        incorrectly_classified_indexes_test = labels_test_by_model != labels_test

        MI_x_train_all = []
        MI_y_train_all = []
        MI_x_test_all = []
        MI_y_test_all = []
        MI_cor_labeled_indexes_all = []
        MI_incor_labeled_indexes_all = []

        for j in range(num_classes):
            #Prepare the data for training and testing attack models (for all data and also correctly labeled samples)
            class_yes_x = confidence_train_for_attack[tuple([labels_train == j])]
            class_no_x = confidence_test_for_attack[tuple([labels_test == j])]

            if class_yes_x.shape[0] < 10 or class_no_x.shape[0] < 10:
                print("Class " + str(j) + " doesn't have enough sample for training an attack model (SKIPPED)!")
                continue

            class_yes_x_correctly_labeled = correctly_classified_indexes_train[tuple([labels_train == j])]
            class_no_x_correctly_labeled = correctly_classified_indexes_test[tuple([labels_test == j])]

            class_yes_x_incorrectly_labeled = incorrectly_classified_indexes_train[tuple([labels_train == j])]
            class_no_x_incorrectly_labeled = incorrectly_classified_indexes_test[tuple([labels_test == j])]

            class_yes_size = int(class_yes_x.shape[0] * what_portion_of_sampels_attacker_knows)
            class_yes_x_train = class_yes_x[:class_yes_size]
            class_yes_y_train = np.ones(class_yes_x_train.shape[0])
            class_yes_x_test = class_yes_x[class_yes_size:]
            class_yes_y_test = np.ones(class_yes_x_test.shape[0])
            class_yes_x_correctly_labeled = class_yes_x_correctly_labeled[class_yes_size:]
            class_yes_x_incorrectly_labeled = class_yes_x_incorrectly_labeled[class_yes_size:]

            class_no_size = int(class_no_x.shape[0] * what_portion_of_sampels_attacker_knows)
            class_no_x_train = class_no_x[:class_no_size]
            class_no_y_train = np.zeros(class_no_x_train.shape[0])
            class_no_x_test = class_no_x[class_no_size:]
            class_no_y_test = np.zeros(class_no_x_test.shape[0])
            class_no_x_correctly_labeled = class_no_x_correctly_labeled[class_no_size:]
            class_no_x_incorrectly_labeled = class_no_x_incorrectly_labeled[class_no_size:]

            y_size = class_yes_x_train.shape[0]
            n_size = class_no_x_train.shape[0]
            if sampling == "undersampling":
                if y_size > n_size:
                    class_yes_x_train = class_yes_x_train[:n_size]
                    class_yes_y_train = class_yes_y_train[:n_size]
                else:
                    class_no_x_train = class_no_x_train[:y_size]
                    class_no_y_train = class_no_y_train[:y_size]
            elif sampling == "oversampling":
                if y_size > n_size:
                    class_no_x_train = np.tile(class_no_x_train, (int(y_size / n_size), 1))
                    class_no_y_train = np.zeros(class_no_x_train.shape[0])
                else:
                    class_yes_x_train = np.tile(class_yes_x_train, (int(n_size / y_size), 1))
                    class_yes_y_train = np.ones(class_yes_x_train.shape[0])

            MI_x_train = np.concatenate((class_yes_x_train, class_no_x_train), axis=0)
            MI_y_train = np.concatenate((class_yes_y_train, class_no_y_train), axis=0)
            MI_x_test = np.concatenate((class_yes_x_test, class_no_x_test), axis=0)
            MI_y_test = np.concatenate((class_yes_y_test, class_no_y_test), axis=0)

            MI_x_train_all.extend(MI_x_train)
            MI_y_train_all.extend(MI_y_train)
            MI_x_test_all.extend(MI_x_test)
            MI_y_test_all.extend(MI_y_test)

            MI_cor_labeled_indexes = np.concatenate((class_yes_x_correctly_labeled, class_no_x_correctly_labeled), axis=0)
            MI_incor_labeled_indexes = np.concatenate((class_yes_x_incorrectly_labeled, class_no_x_incorrectly_labeled), axis=0)

            MI_cor_labeled_indexes_all.extend(MI_cor_labeled_indexes)
            MI_incor_labeled_indexes_all.extend(MI_incor_labeled_indexes)

        MI_x_train_all = np.array(MI_x_train_all)
        MI_y_train_all = np.array(MI_y_train_all)
        MI_x_test_all = np.array(MI_x_test_all)
        MI_y_test_all = np.array(MI_y_test_all)

        #To shuffle the training data:
        shuffle_index = np.random.permutation(MI_x_train_all.shape[0])
        MI_x_train_all = MI_x_train_all[shuffle_index]
        MI_y_train_all = MI_y_train_all[shuffle_index]

        # MI attack
        if args.attack_type == "all":
            attack_model = nn.Sequential(nn.Linear(num_classes * (model_index_counter + 1), 128), nn.ReLU(),
                                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        elif args.attack_type == "aggregated":
            attack_model = nn.Sequential(nn.Linear(num_classes, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                         nn.Linear(64, 1), nn.Sigmoid())
        else:
            print("Attack type is not valid!")
            exit()

        attack_model = attack_model.cuda()
        criterion = nn.BCELoss().cuda()
        optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
        MI_x_train_cuda = torch.from_numpy(MI_x_train_all).float().cuda()
        MI_y_train_cuda = torch.from_numpy(MI_y_train_all).float().cuda()
        MI_x_test_cuda = torch.from_numpy(MI_x_test_all).float().cuda()
        MI_y_test_cuda = torch.from_numpy(MI_y_test_all).float().cuda()

        for ep in range(30):
            y_pred = attack_model(MI_x_train_cuda)
            y_pred = torch.squeeze(y_pred)
            train_loss = criterion(y_pred, MI_y_train_cuda)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        y_pred = attack_model(MI_x_test_cuda).cpu().detach().numpy()
        if y_pred.shape[0] > 0:
            MI_attack_auc = roc_auc_score(MI_y_test_all, y_pred)
        else:
            MI_attack_auc = -1

        #Gap attack
        MI_predicted_y_test_blind = np.zeros((MI_x_test_all.shape[0]))
        MI_predicted_y_test_blind[MI_cor_labeled_indexes_all] = 1
        y_pred = np.array(MI_predicted_y_test_blind)
        if y_pred.shape[0] > 0:
            MI_blind_attack_auc = roc_auc_score(MI_y_test_all, y_pred)
        else:
            MI_blind_attack_auc = -1

        print("---------------------")
        print("Ensemble of", model_index_counter + 1, "models:")
        print("Train/Test accuracy:", str(np.round(acc_train*100, 2)), str(np.round(acc_test*100, 2)))
        print(args.attack_type + " " + args.output_type + "-based MI attack AUC:", MI_attack_auc)
        print("Gap attack AUC:", MI_blind_attack_auc)
        print("---------------------")


if __name__ == '__main__':
    main()


