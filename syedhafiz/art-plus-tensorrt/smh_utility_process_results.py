import numpy as np
from sklearn.metrics import classification_report
to_percentage=100
def process_results(predictions, ground_true):
    predictions_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(ground_true,axis=1)
    # print('{},{}'.format(predictions_classes.shape,predictions_classes[0]))
    # print('{},{}'.format(true_classes.shape,true_classes[0]))
    res_dic = classification_report(true_classes, predictions_classes, output_dict=True)
    acc = res_dic.get("accuracy")
    mac_precision = res_dic.get("macro avg").get("precision")
    mac_recall = res_dic.get("macro avg").get("recall")
    mac_f1_score = res_dic.get("macro avg").get("f1-score")
    support = res_dic.get("macro avg").get("support")
    print('accuracy : {:.2f}; precision : {:.2f}; recall : {:.2f}; f1-score : {:.2f}; support : {} '.format(acc*to_percentage, mac_precision*to_percentage, mac_recall*to_percentage, mac_f1_score*to_percentage, support))#mac_precision)
    