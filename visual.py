import matplotlib.pyplot as plt

# Accuracy values
gender_accuracy = 85.5  # Example value

labels = ['Gender Accuracy',]
accuracies = [gender_accuracy]

plt.bar(labels, accuracies, color=['blue'])
plt.ylim(0, 100)  # Set y-axis limits from 0 to 100
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy')
plt.show()

# from sklearn.metrics import roc_curve, auc

# # Assuming you have predicted probabilities for the positive class
# y_true = [...]  # True binary labels
# y_scores = [...]  # Predicted probabilities for the positive class

# fpr, tpr, _ = roc_curve(y_true, y_scores)
# roc_auc = auc(fpr, tpr)

# plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
# plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
