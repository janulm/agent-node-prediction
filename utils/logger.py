from typing import Callable, List, Optional, Tuple
from typing_extensions import Literal, TypedDict
import matplotlib.pyplot as plt


###
# Logger File:
# should log training, all with validation and test acc and loss
# should also log the model
# should also create a picture of performance
# print out final test acc, loss
###

class HistoryDict(TypedDict):
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]

class Logger:
    def __init__(self, title,job_id, use_acc=False,use_loss=False):
        from typing import Literal, TypedDict
        self.history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'val_acc': [], 'test_loss': [], 'val_loss': []}
        self.title = title
        self.job_id = job_id
        self.use_acc = use_acc
        self.use_loss = use_loss
        assert(use_acc or use_loss)

    def log(p):
        print(p)
        # save data to dict

    def early_stopping(self,min_count_epochs=210,patience=200) -> bool:
        # check if validation acc has increased in the last 100 epochs
        # if not, stop training
        # if yes, reset counter
        if self.use_acc and len(self.history["val_acc"]) > min_count_epochs:
            if self.history["val_acc"][-1] < self.history["val_acc"][-patience]:
                return True

        if self.use_loss and len(self.history["val_loss"]) > min_count_epochs:
            if self.history["val_loss"][-1] > self.history["val_loss"][-patience]:
                return True
        return False

    def log_data(self,train_acc=-1,val_acc=-1,test_acc=-1,test_loss=-1,train_loss=-1,val_loss=-1):
        if self.use_loss:
            self.history['test_loss'].append(test_loss)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
        if self.use_acc:
            self.history['test_acc'].append(test_acc)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)


    def get_test_at_best_val(self):
        # print out test values at highest val values index
        # highest acc or lowest loss? 
        if self.use_acc:
            max_index = self.history["val_acc"].index(max(self.history["val_acc"]))
            val_acc = self.history["val_acc"][max_index]
            test_acc = self.history["test_acc"][max_index]
            return test_acc
        if self.use_loss:
            min_index = self.history["val_loss"].index(min(self.history["val_loss"]))
            val_loss = self.history["val_loss"][min_index]
            test_loss = self.history["test_loss"][min_index]
            return test_loss
    
    def save_plot_data(self):
        if self.use_acc and self.use_loss:
            # create plot
            plt.suptitle(self.title,fontsize=14)
            ax1 = plt.subplot(121)
            ax1.set_title("Loss")
            ax1.plot(self.history["train_loss"], label="train")
            ax1.plot(self.history["val_loss"], label="val")
            plt.xlabel("Epoch")
            ax1.legend()

            ax2 = plt.subplot(122)
            ax2.set_title("Accuracy")
            ax2.plot(self.history["train_acc"], label="train")
            ax2.plot(self.history["val_acc"], label="val")
            plt.xlabel("Epoch")
            ax2.legend()
            plt.savefig("slurm_log/"+str(self.job_id)+".jpg",dpi=300)

            # print out test values at highest val values index
            # highest acc or lowest loss? 
            max_index = self.history["val_acc"].index(max(self.history["val_acc"]))
            val_acc = self.history["val_acc"][max_index]
            val_loss = self.history["val_loss"][max_index]
            test_acc = self.history["test_acc"][max_index]
            test_loss = self.history["test_loss"][max_index]
            print("Max validation acc: ",val_acc, " loss: ",val_loss)
            print("Test acc (same epoch): ",test_acc, " loss: ",test_loss)
        elif self.use_acc:
            # create plot
            plt.suptitle(self.title,fontsize=14)
            ax1 = plt.subplot(111)
            ax1.set_title("Acc")
            ax1.plot(self.history["train_acc"], label="train")
            ax1.plot(self.history["val_acc"], label="val")
            plt.xlabel("Epoch")
            ax1.legend()

            plt.savefig("slurm_log/"+str(self.job_id)+".jpg",dpi=300)

            # print out test values at highest val values index
            # highest acc or lowest loss? 
            max_index = self.history["val_acc"].index(max(self.history["val_acc"]))
            val_acc = self.history["val_acc"][max_index]
            test_acc = self.history["test_acc"][max_index]
            print("Max validation acc: ",val_acc)
            print("Test acc (same epoch): ",test_acc)
        elif self.use_loss:
            # create plot
            plt.suptitle(self.title,fontsize=14)
            ax1 = plt.subplot(111)
            ax1.set_title("Loss")
            ax1.plot(self.history["train_loss"], label="train")
            ax1.plot(self.history["val_loss"], label="val")
            plt.xlabel("Epoch")
            ax1.legend()

            plt.savefig("slurm_log/"+str(self.job_id)+".jpg",dpi=300)

            # print out test values at highest val values index
            # highest acc or lowest loss? 
            min_index = self.history["val_loss"].index(min(self.history["val_loss"]))
            val_loss = self.history["val_loss"][min_index]
            test_loss = self.history["test_loss"][min_index]
            print("Min validation loss: ",val_loss)
            print("Test loss (same epoch): ",test_loss)
        else:
            raise Exception("No data to plot")