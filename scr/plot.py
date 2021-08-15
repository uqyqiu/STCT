import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle


def rocplot(n_classes,target_true,output):
    class_name = ["N","S","V","F","Q"]
    target_true = label_binarize(target_true,classes=[0,1,2,3,4])
    fpr = dict()
    tpr = dict()
    thred={}
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thred[i] = roc_curve(target_true[:,i], output[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw=2
    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','azure','seashell','peru'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of {0} (area = {1:0.4f})' 
                    ''.format(class_name[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    
def plotConMat(Conf_Mat):
    class_name = ["N","S","V","F","Q"]
    plt.imshow(Conf_Mat,cmap=plt.cm.Blues)
    indices = range(len(Conf_Mat))
    plt.xticks(indices, class_name)
    plt.yticks(indices,class_name)
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title("Confusion Matrix")
    plt.show()