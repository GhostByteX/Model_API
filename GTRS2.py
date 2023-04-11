import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import warnings
import pickle

warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class TWC_GTRS_MODEL:
    def __init__(self):
        pass

    def preproc(self, df):
        df.columns = df.columns.str.lower()
        columns_to_drop = ["customer id", "internet service",
                           "gender", "under 30", "senior citizen",
                           "churn category", "churn reason", "customer satisfaction", "unlimited data", "total refunds",
                           "total customer svc requests", "streaming tv", "streaming movies",
                           "product/service issues reported", "premium tech support"
            , "phone service", "online security", "online backup", "offer", "multiple lines", "married",
                           "device protection plan"]
        df.drop(columns_to_drop, inplace=True, axis=1)
        numerical_columns = [column for column in df.columns if df[column].dtype != "object"]
        numerical_columns.remove("churn value")
        categorical_columns = [column for column in df.columns if df[column].dtype == "object"]

        # Extacting top 15 most frequent city values
        top_city = list(df["city"].value_counts().sort_values(ascending=False).head(15).index)

        # Keep city same if it is in top 15, change to "other" otherwise
        df.loc[~df["city"].isin(top_city), "city"] = "other"
        top_city.append("other")
        df["city"].unique()
        X = df.drop(labels=["churn value"], axis=1)
        y = df["churn value"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # Seperating categorical columns for ordinal and onehot encoding (Minimum 10 categories)
        ordinal_columns = [column for column in categorical_columns if df[column].nunique() <= 10]
        onehot_columns = [column for column in categorical_columns if df[column].nunique() > 10]
        # Creating one hot encoder
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        # Applying one hot encoding to categorical columns
        one_hot_cols_train = pd.DataFrame(one_hot_encoder.fit_transform(X_train[onehot_columns]))
        one_hot_cols_test = pd.DataFrame(one_hot_encoder.transform(X_test[onehot_columns]))

        # One hot encoding remove index, putting it back
        one_hot_cols_train.index = X_train.index
        one_hot_cols_test.index = X_test.index

        # Updating names of one hot encoded columns
        c = one_hot_cols_train.columns
        array = ["city_" + column for column in top_city]
        one_hot_cols_train.columns = array
        one_hot_cols_test.columns = array

        # Removing categorical columns (Will replace with one hot encoded columns)
        num_X_train = X_train.drop(onehot_columns, axis=1)
        num_X_test = X_test.drop(onehot_columns, axis=1)

        # Adding one hot encoded columns to datasets
        X_train = pd.concat([num_X_train, one_hot_cols_train], axis=1)
        X_test = pd.concat([num_X_test, one_hot_cols_test], axis=1)
        # X_train.columns
        # Creating ordinal encoder
        ordinal_encoder = OrdinalEncoder(dtype=np.int32)

        # Encoding categorical columns for training data
        X_train[ordinal_columns] = ordinal_encoder.fit_transform(X_train[ordinal_columns])

        # Encoding categorical columns for training data
        X_test[ordinal_columns] = ordinal_encoder.transform(X_test[ordinal_columns])
        X_train
        return df, X_train, X_test, y_train, y_test

    def fit(self, X_train, X_test, y_train, y_test):
        model = TWC_GTRS_MODEL()
        df3 = X_train
        df3['churn value'] = y_train
        df4 = X_test
        df4['churn value'] = y_test
        target_col = 'churn value'
        for col in df3.columns:
            if col != target_col:
                c = df3.groupby(col)[target_col].mean()
                df3['cp_' + col] = df3[col].map(c)
                df4['cp_' + col] = df4[col].map(c)

        df4['final_prob'] = 0
        cp_cols = [col for col in df4.columns if col.startswith('cp_')]
        arithmetic_mean = df4[cp_cols].mean(axis=1) + df4[cp_cols].mean(axis=1) * 0.75
        df4['final_prob'] = arithmetic_mean
        alpha, beta = model.calculate_optimal_eff_cov(df4)

        return df4, alpha, beta

    def calculate_effectiveness_coverage(self, df4, target_col, alpha, beta):
        TOTAL_X = len(df4)
        POS_X = 0
        NEG_X = 0
        BND_X = 0

        final_prob = list(df4['final_prob'])
        pred = []
        for prob in final_prob:
            if prob >= alpha:
                POS_X += 1
                pred.append(1)
            elif prob <= beta:
                NEG_X += 1
                pred.append(0)
            elif prob > beta and prob < alpha:
                BND_X += 1
                pred.append(-1)

        df4['pred'] = pred
        true_churners = len(df4[(df4[target_col] == 1) & (df4['pred'] == 1)])
        true_non_churners = len(df4[(df4[target_col] == 0) & (df4['pred'] == 0)])
        false_non_churners = len(df4[(df4[target_col] == 0) & (df4['pred'] == 1)])
        false_churners = len(df4[(df4[target_col] == 1) & (df4['pred'] == 0)])

        if POS_X >= true_churners:
            term_1 = true_churners
        elif POS_X < true_churners:
            term_1 = POS_X

        if NEG_X >= true_non_churners:
            term_2 = true_non_churners
        elif NEG_X < true_non_churners:
            term_2 = NEG_X

        eff_num = term_1 + term_2

        eff_denom = POS_X + NEG_X

        if eff_denom != 0:
            eff = eff_num / eff_denom
        else:
            eff = 0

        coverage_num = POS_X + NEG_X
        coverage_denom = TOTAL_X
        coverage = coverage_num / coverage_denom

        return eff, coverage

    def calculate_optimal_eff_cov(self, df4):
        model = TWC_GTRS_MODEL()
        desired_effectiveness = 0.99
        desired_coverage = 0.55
        target_col = 'churn value'

        # Set the range of alpha and beta values to explore
        alpha_range = np.arange(0.0, 1.01, 0.01)
        beta_range = np.arange(0.0, 1.01, 0.01)

        # Initialize the best values of alpha and beta and the maximum deviation
        best_alpha = None
        best_beta = None
        max_deviation = np.inf

        # Explore all possible combinations of alpha and beta values
        for alpha in alpha_range:
            for beta in beta_range:
                if alpha > beta:
                    # Calculate the effectiveness and coverage for the current values of alpha and beta
                    effectiveness, coverage = model.calculate_effectiveness_coverage(df4, target_col, alpha, beta)

                    # Calculate the deviation from the desired levels of effectiveness and coverage
                    deviation_effectiveness = abs(effectiveness - desired_effectiveness)
                    deviation_coverage = abs(coverage - desired_coverage)

                    # Calculate the maximum deviation
                    max_deviation_current = max(deviation_effectiveness, deviation_coverage)

                    # If the maximum deviation is smaller than the current maximum deviation, update the best alpha and beta values
                    if max_deviation_current < max_deviation:
                        best_alpha = alpha
                        best_beta = beta
                        max_deviation = max_deviation_current

        return best_alpha, best_beta

    def results_reporting(self, new_df, eff, cov):

        print("Effectivness is : ", eff * 100, "%\n\n")
        print("Coverage is : ", cov * 100, "%\n\n")

        new_df = new_df.drop(index=new_df[new_df == -1].dropna(how='all').index)
        cm = confusion_matrix(new_df['churn value'], new_df['pred'])
        mse = mean_squared_error(new_df['churn value'], new_df['pred'])
        print("Mean Squared Error:", mse, "\n\n")
        accuracy = accuracy_score(new_df['churn value'], new_df['pred'])

        print("Training Accuracy:", accuracy * 100, "%\n\n")
        target_names = ['0', '1']
        print(classification_report(new_df['churn value'], new_df['pred'], target_names=target_names))

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
        cm_display.plot()
        plt.show()

    def main(self):
        model = TWC_GTRS_MODEL()
        df = pd.read_csv('dataset.csv')
        df, X_train, X_test, y_train, y_test = model.preproc(df)
        df4, alpha, beta = model.fit(X_train, X_test, y_train, y_test)
        target_col = 'churn value'
        eff, cov = model.calculate_effectiveness_coverage(df4, target_col, alpha, beta)
        new_df = df4[['churn value', 'pred']]
        new_df = new_df.drop(index=new_df[new_df == -1].dropna(how='all').index)
        model.results_reporting(new_df, eff, cov)

    def predict(self, pred_df):
        model = TWC_GTRS_MODEL()
        pred_df, X, y = model.predict_preproc(pred_df)
        final_prob = model.predict_fit(pred_df, X, y)
        alpha = 0.61
        beta = 0.44
        target_col = 'churn value'
        pred = model.predict_label(final_prob, alpha, beta)
        res = pred
        return res

    def predict_preproc(self, pred_df):
        pred_df.columns = pred_df.columns.str.lower()
        columns_to_drop = ["customer id", "internet service",
                           "gender", "under 30", "senior citizen",
                           "churn category", "churn reason", "customer satisfaction", "unlimited data", "total refunds",
                           "total customer svc requests", "streaming tv", "streaming movies",
                           "product/service issues reported", "premium tech support"
            , "phone service", "online security", "online backup", "offer", "multiple lines", "married",
                           "device protection plan"]
        pred_df.drop(columns_to_drop, inplace=True, axis=1)
        # print(pred_df.info())
        numerical_columns = [column for column in pred_df.columns if pred_df[column].dtype != "object"]
        categorical_columns = [column for column in pred_df.columns if pred_df[column].dtype == "object"]
        top_city = list(pred_df["city"].value_counts().sort_values(ascending=False).head(15).index)

        # Keep city same if it is in top 15, change to "other" otherwise
        pred_df.loc[~pred_df["city"].isin(top_city), "city"] = "other"
        top_city.append("other")
        pred_df["city"].unique()
        X = pred_df.drop(labels=["churn value"], axis=1)
        y = pred_df["churn value"]
        # ordinal_columns = [column for column in categorical_columns if pred_df[column].nunique() <= 10]
        # onehot_columns = [column for column in categorical_columns if pred_df[column].nunique() > 10]
        ordinal_columns = [column for column in categorical_columns if column != "city"]
        onehot_columns = ["city"]
        # Creating one hot encoder
        # one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        with open('encoder.pkl', 'rb') as f:
            one_hot_encoder = pickle.load(f)



        one_hot_cols_test = pd.DataFrame(one_hot_encoder.transform(X[onehot_columns]))
        one_hot_cols_test.index = X.index
        # array = ["city_" + column for column in top_city]
        array = ['city_Los Angeles', 'city_San Diego', 'city_San Jose', 'city_Sacramento', 'city_San Francisco',
                 'city_Fresno', 'city_Long Beach', 'city_Oakland', 'city_Escondido', 'city_Stockton', 'city_Fallbrook',
                 'city_Glendale', 'city_Bakersfield', 'city_Temecula', 'city_Riverside', 'city_other']


        one_hot_cols_test.columns = array
        num_X_test = X.drop(onehot_columns, axis=1)
        X = pd.concat([num_X_test, one_hot_cols_test], axis=1)
        # X_train.columns

        # Creating ordinal encoder
        # ordinal_encoder = OrdinalEncoder(dtype=np.int32)

        with open('ordinal_encoder.pkl', 'rb') as f:
            ordinal_encoder = pickle.load(f)


        X[ordinal_columns] = ordinal_encoder.transform(X[ordinal_columns])

        return pred_df, X, y

    def predict_fit(self, pred_df, X, y):
        model = TWC_GTRS_MODEL()
        df4 = X
        df4['churn value'] = y
        target_col = 'churn value'
        for col in df4.columns:
            if col != target_col:
                c = df4.groupby(col)[target_col].mean()
                df4['cp_' + col] = df4[col].map(c)

        df4['final_prob'] = 0
        cp_cols = [col for col in df4.columns if col.startswith('cp_')]
        arithmetic_mean = (df4[cp_cols].mean(axis=1) + (df4[cp_cols].mean(axis=1) * 0.75)) % 1
        df4['final_prob'] = arithmetic_mean 
        final_prob = df4['final_prob']
        print(final_prob)
        print(df4.head())
        return final_prob

    def predict_label(self, prob, alpha, beta):
        if prob[0] >= alpha:
            return 1
        elif prob[0] <= beta:
            return 0
        elif prob[0] > beta and prob[0] < alpha:
            return -1
