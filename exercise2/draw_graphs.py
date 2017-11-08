import pickle
import matplotlib.pyplot as plt

train_err = []
train_acc = []
val_err = []
val_acc =[]

train_err_pr = []
train_acc_pr = []
val_err_pr = []
val_acc_pr =[]


with open("err_acc.txt", "rb") as fp:
        train_err = pickle.load(fp) 
        train_acc =pickle.load(fp) 
        val_err =pickle.load(fp) 
        val_acc =pickle.load(fp) 


with open("err_acc_pretrained.txt", "rb") as fp:
        train_err_pr = pickle.load(fp) 
        train_acc_pr =pickle.load(fp) 
        val_err_pr =pickle.load(fp) 
        val_acc_pr =pickle.load(fp) 

# plot error
plt.plot(train_err_pr, color='red', label='training error - pre_trained')
plt.plot(train_err, color='green', label='train error - from scratch', linestyle='--')
plt.plot(val_err_pr, color='blue', label='val error - pre_trained')
plt.plot(val_err, color='pink', label='val error - from scratch', linestyle='--')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='best')
plt.savefig('plot_error.png')
plt.close()        

# plot accuracy
plt.plot(train_acc_pr, color='red', label='training accuracy - pre_trained')
plt.plot(train_acc, color='green', label='train accuracy - from scratch', linestyle='--')
plt.plot(val_acc_pr, color='blue', label='val accuracy - pre_trained')
plt.plot(val_acc, color='pink', label='val accuracy - from scratch', linestyle='--')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.savefig('plot_acc.png')
plt.close()    
