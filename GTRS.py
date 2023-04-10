import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class TWC_GTRS_MODEL:
    def __init__(self):
        pass
        
    def predict(self,pred_df):
        model = TWC_GTRS_MODEL()
        #pred_df,X,y = model.predict_preproc(pred_df)
        #final_prob = model.predict_fit(pred_df,X,y)
        final_prob = 0.65
        alpha = 0.61
        beta = 0.44
        target_col = 'churn value'
        pred = model.predict_label(final_prob,alpha,beta)
        
        return pred
    
    
    # def predict_preproc(self,pred_df): 
    #     pred_df.columns = pred_df.columns.str.lower()
    #     columns_to_drop = ["customer id", "internet service",
    #                    "gender", "under 30", "senior citizen",
    #                    "churn category", "churn reason", "customer satisfaction","unlimited data","total refunds",
    #                    "total customer svc requests","streaming tv","streaming movies","product/service issues reported","premium tech support"
    #                   ,"phone service","online security","online backup","offer","multiple lines","married","device protection plan"]
    #     pred_df.drop(columns_to_drop, inplace=True, axis=1)
    #     print(pred_df.info())
    #     numerical_columns = [column for column in pred_df.columns if pred_df[column].dtype != "object"]
    #     categorical_columns = [column for column in pred_df.columns if pred_df[column].dtype == "object"]
    #     top_city = list(pred_df["city"].value_counts().sort_values(ascending=False).head(15).index)

    #     # Keep city same if it is in top 15, change to "other" otherwise
    #     pred_df.loc[~pred_df["city"].isin(top_city), "city"] = "other"
    #     top_city.append("other")
    #     pred_df["city"].unique()
    #     X = pred_df.drop(labels=["churn value"], axis=1)
    #     y = pred_df["churn value"]
    #     ordinal_columns = [column for column in categorical_columns if pred_df[column].nunique() <= 10]
    #     onehot_columns = [column for column in categorical_columns if pred_df[column].nunique() > 10]
    #        # Creating one hot encoder
    #     one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    #     one_hot_cols_test = pd.DataFrame(one_hot_encoder.transform(X[onehot_columns]))
    #     one_hot_cols_test.index = X.index
    #     array = ["city_" + column for column in top_city]
    #     one_hot_cols_test.columns = array
    #     num_X_test = X.drop(onehot_columns, axis=1)
    #     X = pd.concat([num_X_test, one_hot_cols_test], axis=1)
    #     #X_train.columns
    #     # Creating ordinal encoder
    #     ordinal_encoder = OrdinalEncoder(dtype=np.int32)
    #     X[ordinal_columns] = ordinal_encoder.transform(X[ordinal_columns])
          
    #     return df,X,y
    
    # def predict_fit(self,pred_df,X,y):
    #     model = TWC_GTRS_MODEL()
    #     df4 = X
    #     df4['churn value'] = y
    #     target_col = 'churn value'
    #     for col in df4.columns:
    #         if col != target_col:
    #             c = df4.groupby(col)[target_col].mean()
    #             df4['cp_'+col] = df4[col].map(c)

    #     df4['final_prob'] = 0
    #     cp_cols = [col for col in df4.columns if col.startswith('cp_')]
    #     arithmetic_mean = df4[cp_cols].mean(axis=1) + df4[cp_cols].mean(axis=1)*0.75
    #     df4['final_prob'] = arithmetic_mean
    #     final_prob = df4['final prob']
    #     return final_prob
    
    def predict_label(self,prob,alpha,beta):
        if prob >= alpha:
            return 1
        elif prob <= beta:
            return 0
        elif prob > beta and prob < alpha:
            return -1
        
        