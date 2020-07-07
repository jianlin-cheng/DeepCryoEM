function [ acc,sen_recall,spe,pre,TPR,TNR,Miss_class,F_1_score] = evaluation( confusion )
%PERFORMANCE evaluate the performance of prediction 
%   against true labels

% calculate the sensitivity = t-pos/pos  /* true positive recognition rate */
TP=confusion(1);
FN=confusion(2);
FP=confusion(3);
TN=confusion(4);
Total=TP+FN+FP+TN;
 
% Sensitivity or true positive rate (TPR).. sen=TP/(TP+FN)
  sen_recall=TP/(TP+FN);
  
% specificity or true positive rate (TPR).. sen=TP/(TP+FN)
  spe=TP/(TP+FP);

% Precision or positive predictive value (PPV)..prec=TP/(TP+FP)
  pre=TP/(TP+FP);

% True Positive Rate
  TPR=TP/(TP+FP);

% True Negative Rate
if TN==0
    TNR=0;
else
  TNR=TN/(TN+FP);
end

% Misclassification Rate = (FP+FN)/total 
  Miss_class=(FN)/(FN+TP);

% Accuracy
  acc = (TP+TN)/Total;
  
% F1 score  
  F_1_score=(2*TP)/(2*TP+FP+FN);
 
end