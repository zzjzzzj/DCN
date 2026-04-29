# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import importlib
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser, Namespace
from utils.args import add_management_args
from datasets import ContinualDataset
from datasets import get_dataset
from models import get_model
from utils.best_args import best_args
from utils.conf import set_random_seed
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Imports from utils/training.py implementation
from utils.status import progress_bar, create_stash
from utils.tb_logger import TensorboardLogger
from utils.loggers import *
from utils.loggers import CsvLogger
from models.utils.continual_model import ContinualModel
# from datasets.utils.continual_dataset import ContinualDataset # already imported above
from typing import Tuple

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, eval_tea=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    current_model = model.net
    if eval_tea:
        current_model = model.teacher_model
    status = current_model.training
    current_model.eval()

    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            outputs = current_model(inputs)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            #Task incremntal learnning results
            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    current_model.train(status)

    return accs, accs_mask_classes

def visualize_tsne(model: ContinualModel, dataset: ContinualDataset, task_id: int):
    status = model.net.training
    model.net.eval()
    all_features = []
    all_labels = []
    
    # Iterate over all test loaders for tasks seen so far
    print(f"Generating t-SNE for task {task_id}...")
    for k, test_loader in enumerate(dataset.test_loaders):
        if k > task_id:
            break
            
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(model.device)
            
            with torch.no_grad():
                ret = model.net(inputs, return_features=True)
                # ResNet returns: [f0...], linear, eq_head, inv_head, features(normalized)
                if isinstance(ret, (list, tuple)) and len(ret) >= 5:
                     features = ret[4] # Normalized features
                elif isinstance(ret, (list, tuple)):
                     features = ret[-1]
                else:
                     features = ret
                
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    if len(all_features) > 0:
        all_features = np.concatenate(all_features)
        all_labels = np.concatenate(all_labels)
        
        # Subsample if too many points to save time
        if all_features.shape[0] > 2000:
             indices = np.random.choice(all_features.shape[0], 2000, replace=False)
             all_features = all_features[indices]
             all_labels = all_labels[indices]

        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=0)
        try:
            features_2d = tsne.fit_transform(all_features)
            
            plt.figure(figsize=(10, 8))
            unique_labels = np.unique(all_labels)
            
            # Get colormap
            if len(unique_labels) > 10:
                cmap = plt.get_cmap('tab20')
            else:
                cmap = plt.get_cmap('tab10')

            for i, label in enumerate(unique_labels):
                mask = all_labels == label
                # Use i or label for color mapping
                color = cmap(i % 20)
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=str(label), s=10, color=color)
                
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f't-SNE Task {task_id}')
            plt.tight_layout()
            
            save_dir = os.path.join('results', 'tsne', dataset.SETTING, dataset.NAME, model.NAME)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            save_path = os.path.join(save_dir, f'tsne_task_{task_id}.png')
            plt.savefig(save_path)
            plt.close()
            print(f't-SNE plot saved to {save_path}')
        except Exception as e:
            print(f"Failed to run t-SNE: {e}")

    model.net.train(status)


def train_with_tsne(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []
    model_stash = create_stash(model, args, dataset)

    tea_loggers = {}
    tea_results = {}
    tea_results_mask_classes = {}

    if hasattr(model, 'teacher_model'):
        tea_results['teacher_model'], tea_results_mask_classes['teacher_model'] = [], []

    if args.csv_log:
        #csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
        if hasattr(model, 'teacher_model'):
            #print(f'Creating Logger for the teacher model')
            tea_loggers['teacher_model'] = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    # Evaluate teacher model only if present
    if hasattr(model, 'teacher_model'):
        random_results_class, random_results_task = evaluate(model, dataset_copy, eval_tea=True)
    else:
        random_results_class, random_results_task = [], []
    # print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)
                loss = model.observe(inputs, labels, not_aug_inputs)
                progress_bar(i, len(train_loader), epoch, t, loss)
                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)
                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        mean_acc = np.mean(accs, axis=1)
        if not hasattr(model, 'teacher_model'):
            print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)
        
        # Visualise t-SNE here
        visualize_tsne(model, dataset, t)

        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)
        if hasattr(model, 'teacher_model'):
            #print(f'Evaluating teacher_model')
            tea_accs = evaluate(model, dataset, eval_tea=True)
            tea_results['teacher_model'].append(tea_accs[0])
            tea_results_mask_classes['teacher_model'].append(tea_accs[1])
            tea_mean_acc = np.mean(tea_accs, axis=1)
            print_mean_accuracy(tea_mean_acc, t + 1, dataset.SETTING)

        if args.csv_log and hasattr(model, 'teacher_model'):
            tea_loggers['teacher_model'].log(tea_mean_acc)
    if args.csv_log and hasattr(model, 'teacher_model'):
        tea_loggers['teacher_model'].add_fwt(results, random_results_class, results_mask_classes, random_results_task)
        tea_loggers['teacher_model'].add_bwt(tea_results['teacher_model'], tea_results_mask_classes['teacher_model'])
        tea_loggers['teacher_model'].add_forgetting(tea_results['teacher_model'], tea_results_mask_classes['teacher_model'])

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log and hasattr(model, 'teacher_model'):
        tea_loggers['teacher_model'].write(vars(args))


def main():
    parser = ArgumentParser(description='ocdnet', allow_abbrev=False)
    parser.add_argument('--exp_label', type=str, default='2022-11-1')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')

    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    #print(args)
    if args.load_best_args:
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()
        #print(args)
        #exit()
    # print(args)
    if args.seed is not None:
        set_random_seed(args.seed)
    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    if isinstance(dataset, ContinualDataset):
        train_with_tsne(model, dataset, args)

if __name__ == '__main__':
    main()
