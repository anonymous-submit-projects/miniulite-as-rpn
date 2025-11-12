#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from util import *
import pandas as pd 
from datetime import timedelta
import torch.optim as optim
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from util import count_trainable_parameters, measure_inference_speed

#pip install XlsxWriter
#jupyter nbconvert --to script Trainer.ipynb


# In[ ]:


class EarlyStopping:
    def __init__(self, patience=10, mode='max', delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if self.mode == 'min':
            self.sign = 1
        else:  # 'max'
            self.sign = -1

    def step(self, score):
        score = self.sign * score  # transform max into min if necessary

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# In[ ]:


def compute_segmentation_metrics(preds, targets, num_classes, eps=1e-6):
    preds = preds.view(-1).cpu()
    targets = targets.view(-1).cpu()

    dice_total          = 0.0
    miou_total          = 0.0
    iou_valid_classes   = 0
    precision_per_class = []
    recall_per_class    = []
    f1_per_class        = []

    # If it is binary, we only evaluate class 1
    classes_to_eval = [1] if num_classes == 1 else range(num_classes)

    for cls in classes_to_eval:
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).sum().float()
        pred_sum = pred_mask.sum().float()
        target_sum = target_mask.sum().float()
        union = pred_sum + target_sum

        # Dice
        if union > 0:
            dice = (2.0 * intersection + eps) / (union + eps)
            dice_total += dice.item()

        # Io u
        union_iou = (pred_mask | target_mask).sum().float()
        if union_iou > 0:
            iou = (intersection + eps) / (union_iou + eps)
            miou_total += iou.item()
            iou_valid_classes += 1

        # Precision, Recall
        tp = intersection.item()
        fp = (pred_mask & ~target_mask).sum().float().item()
        fn = (~pred_mask & target_mask).sum().float().item()

        if (tp + fp + fn) > 0:
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = (2 * precision * recall) / (precision + recall + eps)

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

    mean_dice = dice_total / len(classes_to_eval)
    mean_iou = miou_total / iou_valid_classes if iou_valid_classes > 0 else 0.0
    mean_precision = np.mean(precision_per_class) if precision_per_class else 0.0
    mean_recall = np.mean(recall_per_class) if recall_per_class else 0.0
    mean_f1 = np.mean(f1_per_class) if f1_per_class else 0.0
    q = mean_iou * mean_dice

    return mean_dice, mean_iou, mean_precision, mean_recall, mean_f1, q


# It calculates image by image and then takes the average.
def compute_iou(preds, masks, num_classes=1, eps=1e-6):
    iou_per_class = [ [] for _ in range(num_classes) ]  # list of lists

    batch_size = preds.size(0)

    for i in range(batch_size):
        pred = preds[i]
        mask = masks[i]

        for cls in range(num_classes):
            pred_cls = (pred == cls).float()
            mask_cls = (mask == cls).float()

            if mask_cls.sum() == 0:
                continue  # does not evaluate missing class in ground truth

            intersection = (pred_cls * mask_cls).sum()
            union = ((pred_cls + mask_cls) > 0).float().sum()
            iou = (intersection + eps) / (union + eps)
            iou_per_class[cls].append(iou)

    # average per class
    class_ious = [
        torch.stack(iou_list).mean()
        for iou_list in iou_per_class
        if len(iou_list) > 0
    ]

    if not class_ious:
        return 0.0

    return torch.stack(class_ious).mean().item()



#LOSS FUNCTION 1 CLASSE
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        # ECB (with logits)
        bce_loss = self.bce(inputs, targets)

        # Sigmoid to convert logits → probabilities
        probs = torch.sigmoid(inputs)

        # Flatten for Dice calculation
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice

        # Thoughtful combination
        loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        return loss



# In[ ]:


class Trainer:

    model         = None
    criterion     = None
    optimizer     = None
    scheduler     = None
    learning_rate = None

    #This class only trains segmentation with 1 class
    #If more is needed, use SemanticTrainer
    num_classes   = 2

    def __init__(self, model_filename=None, model_dir=None, info={}, save_xlsx=False, load_best=True, device=None, rewrite_model=False, 
                 loss_function='BCEDiceLoss'):

        if save_xlsx:
            if model_filename is None:
                raise Exception("model_filename is mandatory when with save_xlsx == True")
        self.save_xlsx = save_xlsx
        self.load_best = load_best
        self.loss_function = loss_function

        #saves the model name and directory
        self.model_filename = model_filename
        if model_dir is None:
            model_dir = model_filename
        self.model_dir = model_dir

        #if at least the model name is passed
        if self.model_filename is not None:
            self.model_file_dir = self.model_dir + "/" + self.model_filename
            self.hist_name = self.model_file_dir.replace('.pth', '.xlsx')
            self.best_path           = self.model_file_dir.replace('.pth', '-best.pth')
            self.last_path           = self.model_file_dir.replace('.pth', '-last.pth')
        else:
            self.model_file_dir = None

        if rewrite_model and self.model_file_dir is not None:
            if os.path.exists(self.hist_name):
                os.remove(self.hist_name)
            if os.path.exists(self.model_file_dir):
                os.remove(self.model_file_dir)
            if os.path.exists(self.best_path):
                os.remove(self.best_path)
            if os.path.exists(self.last_path):
                os.remove(self.last_path)

        #extra information to be saved in xlsx
        self.info = info
        #index of the sample image that will be used
        #to save output during training
        self.sample_img_fixed_index = 0
        #Makes some initializations
        self.create_criterion()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print("Device:",self.device)



    def create_criterion(self):
        self.info['loss_function'] = self.loss_function
        if self.loss_function == 'BCEWithLogitsLoss':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_function == 'BCEDiceLoss':
            self.criterion = BCEDiceLoss(bce_weight=0.5)
        #elif self.loss_function == 'BCEDiceLossMulticlass': #Not yet available here
        #    self.criterion = BCEDiceLossMulticlass(bce_weight=0.5)
        else:
            raise ValueError(f'Loss function {self.loss_function} not found.')
        print("Loss function:", self.loss_function)




    def create_scheduler(self, patience=10, factor=0.5, mode='max'):
        self.info['scheduler'] = "ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5, verbose=True)"
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                      mode=mode, 
                                      patience=patience, 
                                      factor=factor)



    def create_optimizer(self):
        self.info['optimizer'] = f"optim.Adam(self.model.parameters(), lr={self.learning_rate})"
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)




    def train_loop(self, images, masks, epoch):
        outputs     = self.get_model_output(images)

        outputs_s   = outputs.squeeze(1)
        masks_s     = masks.squeeze(1).float()

        loss    = self.criterion(outputs_s, masks_s)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        train_loss = loss.item() * images.size(0)

        return train_loss



    def update_history(self, history, train_loss=None, loss=None, dice=None, miou=None,
                   iou=None, precision=None, recall=None, f1=None, q=None, 
                   elapsed_time=None, images_per_sec=None, started=None):
        if train_loss is not None:
            history["train_loss"].append(train_loss)
        if loss is not None:
            history["loss"].append(loss)
        if dice is not None:
            history["dice"].append(dice)
        if miou is not None:
            history["miou"].append(miou)
        if f1 is not None:
            history["f1"].append(f1)
        if iou is not None:
            history["iou"].append(iou)
        if precision is not None:
            history["precision"].append(precision)
        if recall is not None:
            history["recall"].append(recall)
        if q is not None:
            history["q"].append(q)
        if elapsed_time is not None:
            history["elapsed_time"].append(elapsed_time)
        if images_per_sec is not None:
            history["images_per_sec"].append(images_per_sec)
        if started is not None:
            history["started"].append(started)



    def print_last_history_stats(self):
            def format_line(title, epoch_idx):
                epoch = epoch_idx + 1
                values = {k: self.val_history[k][epoch_idx] for k in self.val_history}
                lr = self.val_history.get("lr", [None] * len(self.val_history["loss"]))[epoch_idx]
                gpu_fps = values.get("GPU_FPS", None)
                cpu_fps = values.get("CPU_FPS", None)
                line = (
                    f"{title}:\n"
                    f" Epoch [{epoch}]"
                    f" - Loss: {values.get('train_loss', float('nan')):.4f}"
                    f" Val Loss: {values.get('loss', float('nan')):.4f}"
                    f" Dice: {values.get('dice', float('nan')):.4f}"
                    f" mIoU: {values.get('miou', float('nan')):.4f}"
                    f" F1-score: {values.get('f1', float('nan')):.4f}"
                    f" IoU: {values.get('iou', float('nan')):.4f}"
                    f" Precision: {values.get('precision', float('nan')):.4f}"
                    f" Recall: {values.get('recall', float('nan')):.4f}"
                    f" Q: {values.get('q', float('nan')):.4f}"
                    f" Tempo total: {values.get('elapsed_time', 'nan')}"
                )
                if lr is not None:
                    line += f" LR:{lr:.6f}"
                if gpu_fps is not None:
                    line += f" GPU_FPS: {gpu_fps:.2f}"
                if cpu_fps is not None:
                    line += f" CPU_FPS: {cpu_fps:.2f}"
                return line

            # best time (highest Dice)
            best_epoch = int(max(range(len(self.val_history["dice"])), key=lambda i: self.val_history["dice"][i]))
            print(format_line("Best model", best_epoch))

            # last season
            last_epoch = len(self.val_history["dice"]) - 1
            print(format_line("Latest model", last_epoch))


    def do_save_xlsx(self):

        avg_speed = sum(self.val_history['images_per_sec']) / len(self.val_history['images_per_sec'])
        self.info['training_speed_img_per_sec'] = round(avg_speed, 2)

        df_val_history = pd.DataFrame(self.val_history)
        df_val_history.insert(0, 'epoch', range(1, len(df_val_history)+1))
        df_val_history['epoch'] = df_val_history['epoch'].astype(str)


        df_test_history = pd.DataFrame(self.test_history)
        df_test_history.insert(0, 'epoch', range(1, len(df_test_history)+1))
        df_test_history['epoch'] = df_test_history['epoch'].astype(str)


        df_info = pd.DataFrame(self.info, index=[0])
        with pd.ExcelWriter(self.hist_name, engine='xlsxwriter') as writer:
            df_val_history.to_excel(writer, sheet_name='val_history', index=False, float_format="%.4f")
            df_test_history.to_excel(writer, sheet_name='test_history', index=False, float_format="%.4f")
            df_info.to_excel(writer, sheet_name='model_info', index=False, float_format="%.4f")

            workbook  = writer.book
            worksheet = writer.sheets['val_history']

            chart = workbook.add_chart({'type': 'line'})

            # The 'epoch' column is now in column 0
            # Assuming 'val_dice' is in column 5 and 'val_IoU' in 6 (or adjust this dynamically)
            col_dice = df_val_history.columns.get_loc('dice')
            col_iou  = df_val_history.columns.get_loc('miou')

            chart.add_series({
                'name':       'dice',
                'categories': ['val_history', 1, 0, len(df_val_history), 0],  # column 0 = epoch
                'values':     ['val_history', 1, col_dice, len(df_val_history), col_dice],
            })
            chart.add_series({
                'name':       'mIoU',
                'categories': ['val_history', 1, 0, len(df_val_history), 0],
                'values':     ['val_history', 1, col_iou, len(df_val_history), col_iou],
            })

            chart.set_title({'name': 'Training'})

            chart.set_x_axis({
                'name': 'Epoch',
                'interval_unit': 10,
                'num_font': {'rotation': -45},
            })
            chart.set_y_axis({'name': 'Value'})

            worksheet.insert_chart('K2', chart)



    def load_xlsx_history(self):
        # Read all sheets in the file
        xls = pd.read_excel(self.hist_name, sheet_name=None)

        # Retrieves the history DataFrame and converts it to a dictionary list
        df_val_history   = xls['val_history']
        last_epoch   = int(df_val_history['epoch'].iloc[-1])
        self.val_history = df_val_history.drop(columns=['epoch']).to_dict(orient='list')


        df_test_history   = xls['test_history']
        self.test_history = df_test_history.drop(columns=['epoch']).to_dict(orient='list')

        # Accumulated time
        elapsed_str      = df_val_history['elapsed_time'].iloc[-1]
        h, m, s          = map(int, elapsed_str.split(':'))
        accumulated_time = timedelta(hours=h, minutes=m, seconds=s).total_seconds()
        start_time       = time.time() - accumulated_time  # Adjusts to maintain accumulated count

        # Retrieves model information DataFrame and converts to dictionary
        df_info = xls['model_info']
        self.info = df_info.iloc[0].to_dict()
        return last_epoch, start_time

    def load_model(self, model_file_dir, model=None, load_xlsx=True, load_scheduler=False):
        #if the model is passed
        if model is not None:
            #self.model receives the new model
            self.model = model
        #if the model to be loaded has not been passed
        if self.model is None:
            raise Exception("You need to pass the model object in the 'model' parameter")

        if self.optimizer is None:
            self.create_optimizer()
        if self.scheduler is None:
            self.create_scheduler()

        #loads the model from the .pth file
        checkpoint = torch.load(model_file_dir, weights_only=False)
        #retrieves the states of the file
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_score  = checkpoint['best_acc']
        epoch       = checkpoint['epoch'] + 1
        self.model.to(self.get_device())
        print(f"Loaded model: {model_file_dir}")
        if load_xlsx:
            start_epoch, start_time = self.load_xlsx_history()
            return best_score, epoch, start_epoch, start_time
        return best_score, epoch

    def save_model(self, path, epoch, best_score):
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_acc': best_score
                }, path)

    def get_device(self):
        return self.device


    def get_model_output(self,images):
        return self.model(images)


    def val_loop(self, images, masks):
        outputs     = self.get_model_output(images)
        loss        = self.criterion(outputs, masks)
        val_loss    = loss.item() * images.size(0)

        #make the threshold
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).long()
        masks = masks.long()

        #compute the metrics
        dice, mIoU, precision, recall, f1, q = compute_segmentation_metrics(preds, masks, num_classes=self.num_classes)
        IoU = compute_iou(preds, masks, num_classes=self.num_classes)
        val_dice      = dice      * images.size(0)
        val_mIoU      = mIoU      * images.size(0)
        val_IoU       = IoU       * images.size(0)
        val_precision = precision * images.size(0)
        val_recall    = recall    * images.size(0)
        val_f1        = f1        * images.size(0)
        val_q         = q         * images.size(0)
        return val_loss, val_dice, val_mIoU, val_IoU, val_precision, val_recall, val_f1, val_q

    def evaluate_model(self, data_loader, print_stats=False, model=None):
        if model is not None:
            self.model = model
        self.model.eval()
        device           = self.get_device()
        loss        = 0.0
        dice        = 0.0
        mIoU        = 0.0
        IoU         = 0.0
        precision   = 0.0
        recall      = 0.0
        f1          = 0.0
        q           = 0.0
        with torch.no_grad():
            for images, masks in data_loader:
                images = images.to(device)
                masks  = masks.to(device)
                iloss, idice, imIoU, iIoU, iprecision, irecall, if1, iq = self.val_loop(images, masks)

                loss      += iloss
                dice      += idice
                mIoU      += imIoU
                IoU       += iIoU
                precision += iprecision
                recall    += irecall
                f1        += if1
                q         += iq

        avg_loss        = loss / len(data_loader.dataset)
        avg_dice        = dice / len(data_loader.dataset)
        avg_mIoU        = mIoU / len(data_loader.dataset)
        avg_IoU         = IoU  / len(data_loader.dataset)
        avg_precision   = precision   / len(data_loader.dataset)
        avg_recall      = recall   / len(data_loader.dataset)
        avg_f1          = f1    / len(data_loader.dataset)
        avg_q           = q    / len(data_loader.dataset)

        if print_stats:
            stats = (f"Loss: {avg_loss:.4f} " 
                    f"Dice: {avg_dice:.4f} mIoU: {avg_mIoU:.4f} F1: {avg_f1:.4f}  IoU: {avg_IoU:.4f} " 
                    f"Prec: {avg_precision:.4f} " 
                    f"Recall: {avg_recall:.4f} Q: {avg_q:.4f} ")
            print(stats)

        return avg_loss, avg_dice, avg_mIoU, avg_IoU, avg_precision, avg_recall, avg_f1, avg_q



    def train(self, model, 
                    train_loader, 
                    val_loader,
                    test_loader, 
                    num_epochs=50, 
                    #saves the model every
                    save_every=None,
                    #prints the tempo every
                    print_every=None,
                    #continue training where you left off
                    continue_from_last=False,
                    #verbose==1 prints the training on the same line
                    verbose=3,
                    learning_rate=1e-4,
                    # scheduler patience=decreases IR after 10 epochs without improvement in acc
                    scheduler_patience=10,
                    # early_stop_patience=ends training after 20 epochs if acc improves
                    early_stop_patience=20,
                    measure_cpu_speed=True
                    ):

        torch.backends.cudnn.benchmark = True
        device = self.get_device()

        self.learning_rate  = learning_rate
        self.model          = model
        start_epoch         = 0
        best_score          = -1.0
        best_stats          = ""
        start_time          = time.time()
        started             = False
        batch_size          = train_loader.batch_size

        trainable_parameters = count_trainable_parameters(model)
        print("trainable_parameters:", trainable_parameters)
        self.info['dataset_name']         = train_loader.dataset.__module__
        self.info['dataset_batch_size']   = batch_size
        self.info['trainable_parameters'] = trainable_parameters
        images, labels = next(iter(train_loader))
        self.info['dataset_resolution']   = f"{images.shape[2]} x {images.shape[3]}"


        self.val_history = {
            "train_loss":     [],
            "loss":           [],
            "dice":           [],
            "miou":           [],
            "f1":             [],
            "iou":            [],
            "precision":      [],
            "recall":         [],
            "q":              [],
            "elapsed_time":   [],
            "images_per_sec": [],
            "started":        [],
        }
        self.test_history = {k: [] for k in self.val_history}


        #prints everything on the same line
        tqdm_disable = print_every!=None
        print_end    = '\r\n'
        if verbose == 1:
            print_end    = '\r'
            tqdm_disable = True



        #if the model name was passed
        if self.model_filename is not None:
            #create directories
            os.makedirs(self.model_dir, exist_ok=True)

            #First, it checks whether the final, trained model already exists
            if os.path.exists(self.model_file_dir):
                if self.load_best:
                    #if it already exists, load and return
                    print("Trained model already exists (Loading better version).")
                    self.load_model(self.best_path)
                else:
                    #if it already exists, load and return
                    print("Trained model already exists (Loading latest version).")
                    self.load_model(self.model_file_dir)
                self.print_last_history_stats()
                return model
            #if it does not exist and is a continuation of the training
            elif continue_from_last == True:
                #continues from -last
                if os.path.exists(self.last_path):
                    _, _, start_epoch, start_time = self.load_model(self.last_path)
                    print(f"Continuing from the saved model: {self.last_path}")
                    print(f"start_epoch: {start_epoch}, start_time: {start_time}")
                    if start_epoch >= num_epochs:
                        self.print_last_history_stats()
                        return self.model



        model.to(device)
        self.create_optimizer()
        self.create_scheduler(patience=scheduler_patience)
        early_stopper = EarlyStopping(patience=early_stop_patience, mode='max')



        ## Training
        epoch = start_epoch
        for epoch in range(start_epoch, num_epochs):

            model.train()
            train_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", disable=tqdm_disable):
                images = images.to(device)
                masks  = masks.to(device)
                ## training loop
                train_loss += self.train_loop(images, masks, epoch)

            avg_train_loss = train_loss / len(train_loader.dataset)


            ## Validation
            avg_val_loss, avg_val_dice, avg_val_mIoU, avg_val_IoU, avg_val_precision, avg_val_recall, avg_val_f1, avg_val_q = self.evaluate_model(val_loader)


            ##Test
            avg_test_loss, avg_test_dice, avg_test_mIoU, avg_test_IoU, avg_test_precision, avg_test_recall, avg_test_f1, avg_test_q = self.evaluate_model(test_loader)

            elapsed     = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            current_lr  = self.optimizer.param_groups[0]['lr']
            stats = (f"Epoch [{epoch+1}/{num_epochs}] - " 
                    f"Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} " 
                    f"Dice: {avg_val_dice:.4f} mIoU: {avg_val_mIoU:.4f} F1: {avg_val_f1:.4f} IoU: {avg_val_IoU:.4f} " 
                    f"Prec: {avg_val_precision:.4f} " 
                    f"Recall: {avg_val_recall:.4f} Q: {avg_val_q:.4f} " 
                    f"Time: {elapsed_str} LR:{current_lr:.6f}")


            if print_every is None:
                print(stats,end=print_end)
            else:
                if (epoch+1) % print_every == 0:
                    print(stats,end=print_end)


            images_per_sec = (len(train_loader) * batch_size) / elapsed

            ## Saves the evolution of the network
            self.update_history(
                self.val_history,
                train_loss=avg_train_loss,
                loss=avg_val_loss,
                dice=avg_val_dice,
                miou=avg_val_mIoU,
                iou=avg_val_IoU,
                precision=avg_val_precision,
                recall=avg_val_recall,
                f1=avg_val_f1,
                q=avg_val_q,
                elapsed_time=elapsed_str,
                images_per_sec=images_per_sec,
                started=('started' if not started else '')
            )
            self.update_history(
                self.test_history,
                train_loss=avg_train_loss,
                loss=avg_test_loss,
                dice=avg_test_dice,
                miou=avg_test_mIoU,
                iou=avg_test_IoU,
                precision=avg_test_precision,
                recall=avg_test_recall,
                f1=avg_test_f1,
                q=avg_test_q,
                elapsed_time=elapsed_str,
                images_per_sec=images_per_sec,
                started=('started' if not started else '')
            )




            started = True


            # The avg_val_dice will be observed for the scheduler and early_stopper

            # reduces the learning rate if the score does not improve
            self.scheduler.step(avg_val_dice)

            # for training if you don't improve in X times
            early_stopper.step(avg_val_dice)
            if early_stopper.early_stop:
                print(f"Stopping at epoch {epoch+1} by early stopping.")
                break

            ## Save the best model so far
            if avg_val_dice > best_score:
                best_score = avg_val_dice

                if self.model_file_dir is not None:
                    #save the model at the best time
                    self.save_model(self.best_path, epoch, best_score)
                    current_lr  = self.optimizer.param_groups[0]['lr']
                    best_stats = (f"Epoch [{epoch+1}/{num_epochs}] - " 
                                f"Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} " 
                                f"Dice: {avg_val_dice:.4f} mIoU: {avg_val_mIoU:.4f} F1: {avg_val_f1:.4f} IoU: {avg_val_IoU:.4f} " 
                                f"Prec: {avg_val_precision:.4f} " 
                                f"Recall: {avg_val_recall:.4f} Q: {avg_val_q:.4f} " 
                                f"Time: {elapsed_str} LR:{current_lr:.6f}")
                    if print_every is None and verbose > 1:
                        print("✔ Best model saved:", best_stats, end=print_end)
                    #save excel to the current moment
                    if self.save_xlsx:
                        self.do_save_xlsx()



            #Saves the network every
            if save_every is not None and (epoch + 1) % save_every == 0:
                last_model_file_dir = self.model_file_dir.replace('.pth','-last.pth')
                self.save_model(last_model_file_dir, epoch, best_score)
                self.do_save_xlsx()
                if verbose > 1:
                    print("Saved last as", last_model_file_dir, end=print_end)



        last_stats = (f"Epoch [{epoch+1}/{num_epochs}] - " 
                    f"Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} " 
                    f"Dice: {avg_val_dice:.4f} mIoU: {avg_val_mIoU:.4f} F1: {avg_val_f1:.4f} IoU: {avg_val_IoU:.4f} " 
                    f"Prec: {avg_val_precision:.4f} " 
                    f"Recall: {avg_val_recall:.4f} Q: {avg_val_q:.4f} " 
                    f"Time: {elapsed_str} LR:{current_lr:.6f}")



        #calculates the FPS of the model
        self.info['GPU_FPS'], self.info['GPU_time_per_image'], self.info['CPU_FPS'], self.info['CPU_time_per_image'] = measure_inference_speed(self.model, 
                                                                                                                                               val_loader, 
                                                                                                                                               measure_cpu_speed=measure_cpu_speed)

        print("")
        if best_stats:
            print("Best model:\r\n", best_stats)
        print("Latest model:\r\n", last_stats + '\r\n GPU_FPS:',self.info['GPU_FPS'], ' CPU_FPS:',self.info['CPU_FPS'])


        if self.model_file_dir is not None:
            self.save_model(self.model_file_dir, epoch, best_score)
            print("Saved as", self.model_file_dir)


        if self.save_xlsx:
            # Write the excel file with history
            self.do_save_xlsx()

        #beep win
        #os.system('powershell.exe -Command "[console]::beep(600,200); [console]::beep(600,200);"')
        #linux
        os.system('play -nq -t alsa synth 0.2 sine 600; play -nq -t alsa synth 0.2 sine 600')
        return model


if __name__ == '__main__':
    pass


# In[ ]:


def check_masks_for_ce_loss(masks, num_classes=3, ignore_index=255):
    """
    Checks a mask for invalid values ​​before CrossEntropyLoss.

    masks: tensor [B,H,W] ou [B,1,H,W]
    """
    if masks.ndim == 4:
        masks = masks.squeeze(1)  # [b,h,w]

    invalid_mask = (masks < 0) | ((masks >= num_classes) & (masks != ignore_index))
    has_invalid = invalid_mask.any()

    if has_invalid:
        print("Invalid values ​​found in masks!")
        for b in range(masks.size(0)):
            unique_vals = torch.unique(masks[b])
            if ((unique_vals >= num_classes) & (unique_vals != ignore_index)).any() or (unique_vals < 0).any():
                print(f"Batch {b}: unique values -> {unique_vals.tolist()}")
    else:
        print("All masks valid for CrossEntropyLoss.")


class MulticlassTrainer(Trainer):

    def __init__(self, num_classes, model_filename=None, model_dir=None, info={}, save_xlsx=False, loss_function='CrossEntropyLoss'):
        #Correct the loss_function if necessary
        if loss_function == 'BCEWithLogitsLoss':
            loss_function = 'CrossEntropyLoss'

        super(MulticlassTrainer, self).__init__(model_filename=model_filename, model_dir=model_dir, info=info, save_xlsx=save_xlsx, loss_function=loss_function)
        self.num_classes = num_classes


    def create_criterion(self):

        if self.loss_function == 'CrossEntropyLoss':
            self.info['loss_function'] = 'CrossEntropyLoss'
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        else:
            raise ValueError(f'Loss function {self.loss_function} not found.') 



    def train_loop(self, images, masks, epoch):
        outputs     = self.get_model_output(images)

        masks_s     = masks.long().squeeze(1)

        loss    = self.criterion(outputs, masks_s)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        train_loss = loss.item() * images.size(0)

        return train_loss


    def val_loop(self, images, masks):
        outputs     = self.get_model_output(images)

        masks_s     = masks.long().squeeze(1)


        try:
            loss        = self.criterion(outputs, masks_s)
            val_loss    = loss.item() * images.size(0)
        except Exception as e:
            check_masks_for_ce_loss(masks_s, num_classes=self.num_classes, ignore_index=255)
            raise

        preds       = torch.argmax(outputs, dim=1)
        dice, mIoU, precision, recall, f1, q = compute_segmentation_metrics(preds, masks, self.num_classes)
        IoU = compute_iou(preds, masks, num_classes=self.num_classes)

        val_dice      = dice      * images.size(0)
        val_mIoU      = mIoU      * images.size(0)
        val_IoU       = IoU       * images.size(0)
        val_precision = precision * images.size(0)
        val_recall    = recall    * images.size(0)
        val_f1        = f1        * images.size(0)
        val_q         = q         * images.size(0)

        return val_loss, val_dice, val_mIoU, val_IoU, val_precision, val_recall, val_f1, val_q



