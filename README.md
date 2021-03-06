# Bank-Institution-Term-Deposit-Prediction
Implementation of a machine learning algorithm to predict if a customer to a bank will subscribe to term deposit.

### Goal
The objective is to implement a classifier with a high accuracy that the bank can use to develop a more efficient campaign strategy that will guarantee the subscription of more customers to term deposit hence generating more profits for the bank while minimising the cost spent on the campaign.

### Dataset
1) The data is related with direct marketing campaigns of a bank. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
2) The target or class is the subscribed column which shows the outcome of the campaign with a ‘yes’ or ‘no’ hence, the predictor variable showing if a client subscribed for a term deposit or not.
### Methods 
Built 3 classification model in Python:
- Logistic Regression
- XGBoost
- Multi-Layer Perceptron

### Result
Choosing the right model is always a difficult task to do. Since our target class is imbalanced accuracy might not be the right metric to evaluate our model. In this case, It is better to use F1-score as our metrics. Depending on this we can say our XGBoost Classifier Model performed better than the other two.
