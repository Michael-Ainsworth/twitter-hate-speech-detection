from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def binary_metrics(y_test, y_pred, y_pred_rounded):
    print(confusion_matrix(y_test, y_pred_rounded))
    print(classification_report(y_test, y_pred_rounded))

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_value = roc_auc_score(y_test, y_pred)

    precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
    pr_no_skill = 1 - sum(y_pred_rounded) / len(y_pred_rounded)

    plt.figure(figsize = (14,7))
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, color='darkorange',
            lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(['AUC = {}'.format(auc_value)], loc="lower right")

    plt.subplot(1,2,2)
    plt.plot(recalls, precisions, color='darkorange',
            lw=2)
    plt.plot([0, 1], [pr_no_skill,pr_no_skill], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()