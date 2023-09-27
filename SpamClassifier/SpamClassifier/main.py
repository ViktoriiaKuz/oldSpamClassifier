"""f"""
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import dask.dataframe as dd
import matplotlib.pyplot as plt
import fastparquet
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
import warnings
import spacy
import langdetect
import pickle
import shap
import time
from datetime import datetime
from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidLength
from langdetect import LangDetectException
from sklearn.svm import SVC
from tqdm import tqdm
from settings import RAW_DATA_PATH
from settings import SEPARATED_DATA_PATH
from settings import ROOT_PATH
from settings import PROCESSED_DATA_PATH
from settings import NORMALIZED_DATA_PATH
from pandas.errors import SettingWithCopyWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# Ignore the warning about setting a value on a copy of a slice from a DataFrame
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

LABEL_KEY = 'label'
CHAT_KEY = "ad_chat/ad_trigger/user_to_user"
PHRASES_TO_DELETE = [
    "посмотрел(а) ваш номер телефона",
    "актуально",
    "просмотрел(а) ваш номер телефона",
    "ваше объявление",
    "Name:",
    "bottom.payload",
    "Name: , dtype: object",
    "Name: , Length: , dtype: object",
    "Name: , dtype: object",
    "добавил(а) в Избранное", "dtype:", "object", "dtype: object", "просмотрел(а) ваш номер телефона",
    "Length: , dtype: object", "Ещё актуально?", "n/", "name", "Name", ":"]
ROWS_TO_SEPARATE = 50000
UNKNOWN_KEY = 'Unknown'
RUSSIAN_CODE = 'ru'
ENGLISH_CODE = 'en'
POLISH_CODE = 'pl'
CONVERSATION_TYPE_COL = 'conversartion_type'
OWNER_COL = 'owner'
RECIPIENT_COL = 'recipient'
CHAT_ID_COL = 'chat_id'
AD_TRIGGER_COL = '{ad_trigger,'
AD_CHAT_COL = '{ad_chat,'
USER_TO_USER_COL = '{user_to_user'
UPDATED_COL = 'updated'
MESSAGES_COL = 'user_messages'
DURATION = 'duration'
MONGO_ID_COL = '_id'

RUSSIAN_MODEL = spacy.load('ru_core_news_sm')
POLISH_MODEL = spacy.load('pl_core_news_sm')
ENGLISH_MODEL = spacy.load('en_core_web_sm')

dfs_list = []
newtext = ["Free entry"]
header_text = ['owner', 'total_recipients', 'total_chats', 'newest_message', 'oldest_message', 'label', 'text', 'duration',
          'chats_per_day', 'recipients_per_day', 'language']
header= ['owner', 'total_recipients', 'total_chats', 'newest_message', 'oldest_message', 'label', 'duration',
           'chats_per_day', 'recipients_per_day', 'text', 'language']


def files_separation(file_paths):
    marker = os.path.basename(file_paths).split('_')[0]
    df = pd.read_csv(file_paths, encoding="utf-8-sig")
    list_df = [df[i:i + ROWS_TO_SEPARATE] for i in range(0, len(df), ROWS_TO_SEPARATE)]
    for i, df in tqdm(enumerate(list_df)):
        df[LABEL_KEY] = marker
        df.to_parquet(os.path.join(SEPARATED_DATA_PATH, f"{marker}_{i}.parquet"), index=False)


def replacing_commas(file_path):
    df = pd.read_parquet(file_path)
    df.replace({r'[\n]+': ''}, regex=True).dropna().reset_index(drop=True)
    return df


def parse_values(set_string):
    values_list = set_string[0:-1].split(' ')

    if USER_TO_USER_COL in set_string:
        owner = values_list[0]
        recipient = values_list[3]
        chat_id = 'unknown'
    elif any([t in set_string for t in [AD_CHAT_COL, AD_TRIGGER_COL]]):
        owner = values_list[0]
        recipient = values_list[4]
        chat_id = values_list[2]
    else:
        raise Exception(f"{set_string} contains unknown value")

    return {
        CONVERSATION_TYPE_COL: values_list[1],
        OWNER_COL: owner,
        RECIPIENT_COL: recipient,
        CHAT_ID_COL: chat_id
    }


def separating_colummns(df):
    df[MONGO_ID_COL] = df[MONGO_ID_COL].map(lambda x: parse_values(x))

    for key in OWNER_COL, RECIPIENT_COL, CHAT_ID_COL, CONVERSATION_TYPE_COL:
        df[key] = df[MONGO_ID_COL].map(lambda x: x[key])

    conversation_type_df = pd.get_dummies(df[CONVERSATION_TYPE_COL])
    df.drop(columns=[MONGO_ID_COL, CONVERSATION_TYPE_COL], inplace=True)

    return pd.concat([df, conversation_type_df], axis=1)


def grouping_by(df):
    df = pd.DataFrame(df)
    df['label'] = df['label'].apply(lambda x: 1 if x == "blocked" else 0)
    if 'bottom.payload' not in df.columns:
        df = df.rename(columns={'user_messages': 'bottom.payload'})

    df = (
        df.assign(owner_index=df.index)
        .groupby(OWNER_COL)
        .agg(
            label=(LABEL_KEY, "first"),
            total_recipients=(RECIPIENT_COL, "count"),
            total_chats=(CHAT_ID_COL, "count"),
            newest_message=(UPDATED_COL, "max"),
            oldest_message=(UPDATED_COL, "min"),
            user_messages=("bottom.payload", lambda x: "".join([str(e) for e in x]))
        )
        .reset_index()
    )
    if USER_TO_USER_COL in df:
        df['total_user_to_user'] = df.groupby(USER_TO_USER_COL)["sum"]
    if AD_CHAT_COL in df:
        df['total_ad_chats'] = df.groupby(AD_CHAT_COL)["sum"]
    if AD_TRIGGER_COL in df:
        df['total_ad_trigger'] = df.groupby(AD_TRIGGER_COL)["sum"]

    return df


def duration_processing(df):
    def calculate_duration(first_message_date, last_message_date):
        FORMAT = "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(str(first_message_date), FORMAT) - \
            datetime.strptime(str(last_message_date), FORMAT)

    FIRST_MESSAGE = 'newest_message'
    LAST_MESSAGE = 'oldest_message'
    df['duration'] = list(map(lambda f, l: calculate_duration(f, l), df[FIRST_MESSAGE], df[LAST_MESSAGE]))
    return df


def calculate_total(duration, total):
    if duration.total_seconds() > 0:
        return total / duration.total_seconds()
    else:
        return 0


def total_chats_count(df):
    CHATS_TOTAL = 'total_chats'
    df['chats_per_day'] = list(map(lambda d, t: calculate_total(d, t), df[DURATION], df[CHATS_TOTAL]))
    return df


def total_recipients_count(df):
    RECIPIENTS_TOTAL = 'total_recipients'
    df['recipients_per_day'] = list(map(lambda d, t: calculate_total(d, t), df[DURATION], df[RECIPIENTS_TOTAL]))
    return df


def preprocess_single_text(text):
    nlp = RUSSIAN_MODEL
    if len(text.split()) < 15 or len(text) > 5000:
        lang = "unknown"
        processed_text = text
    else:
        try:
            lang = langdetect.detect(" ".join(text.split(" ")[:15]))
            if lang == RUSSIAN_CODE:
                nlp = RUSSIAN_MODEL
            elif lang == POLISH_CODE:
                nlp = POLISH_MODEL
            elif lang == ENGLISH_CODE:
                nlp = ENGLISH_MODEL

            processed_text = " ".join([token.lemma_ for token in nlp(text)])
        except NotValidLength as ntl:
            lang = "unknown"
            processed_text = text

        except langdetect.lang_detect_exception.LangDetectException:

            translator = GoogleTranslator()
            translation = translator.translate(text, target='en')
            if translation is not None:
                lang = 'translated'
                nlp = spacy.load('en_core_web_sm')
                processed_text = " ".join([token.lemma_ for token in nlp(translation)])
            else:

                lang = 'unknown'
                processed_text = text

    return {
        'text': processed_text,
        'language': lang
    }


def concatenate(df):
    dfs_list.append(df)
    concatenated_df = dd.concat(dfs_list, ignore_index=True)
    concatenated_df.rename(columns={'1': 'label'})
    with open("pickle.pkl", "wb") as f:
        pickle.dump(concatenated_df, f)
    return concatenated_df


def lemmatize(df):
    df = pd.DataFrame(df)
    df.columns = header_text[:10]
    if len(df['text']) > 50000:
        chunk_size = 500
        total_rows = len(df['text'])
        num_chunks = total_rows // chunk_size

        for i in tqdm(range(num_chunks + 1)):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            process_chunk_and_save(df[start_idx:end_idx], i)
    else:
        process_chunk_and_save(df, 1)
    return df


def process_chunk_and_save(chunk_df, i):
    processed_chunk = []

    for text in chunk_df['text']:
        result = preprocess_single_text(text)
        processed_chunk.append(result)

    processed_df = pd.DataFrame(processed_chunk)
    processed_df = processed_df.reset_index(drop=True)
    chunk_df = chunk_df.drop(columns='text')
    processed_df = pd.concat([chunk_df, processed_df], axis=1, ignore_index=True)
    processed_df.to_csv(os.path.join(PROCESSED_DATA_PATH, f'preprocessed_data_{i}.csv'),
                        encoding='utf-8-sig',
                        index=False, errors='ignore')


def drop_columns(df):
    columns_to_drop = ['newest_message', 'oldest_message']
    columns_to_drop_existing = []

    for col in columns_to_drop:
        if col in df.columns:
            columns_to_drop_existing.append(col)

    if columns_to_drop_existing:
        df = df.drop(columns=columns_to_drop_existing)

    return df


def label_dummy_coding(df):
    return df


def numerical(df):
    numerical_df = df.drop(columns=['text', 'label'])
    return numerical_df


def normalize_numerical(df):
    scaler = MinMaxScaler()
    df = df.drop(columns=['duration', 'language', 'owner'])
    # df = df.replace([np.inf, -np.inf], np.nan)
    dft = scaler.fit_transform(df)
    dft = pd.DataFrame(dft)
    return dft


# def text_df_count(df):
#    text_spam = df[df['label'] == 'blocked']['text']
#    text_normal = df[df['label'] == 'regular']['text']

#    blocked_dict = {}
#    normal_dict = {}

#    for text in text_spam:
#        words = text.split()
#        for word in words:
#            blocked_dict[word] = blocked_dict.get(word, 0) + 1

#    for text in text_normal:
#        words = text.split()
#        for word in words:
#            normal_dict[word] = normal_dict.get(word, 0) + 1

#    common_words = set(blocked_dict.keys()) & set(normal_dict.keys())
#    common_words = pd.DataFrame(common_words)

#    blocked_dict = {word: freq for word, freq in blocked_dict.items() if word not in common_words}
#    normal_dict = {word: freq for word, freq in normal_dict.items() if word not in common_words}

#    blocked_df = pd.DataFrame(list(blocked_dict.items()), columns=['word', 'spam_frequency'])
#    normal_df = pd.DataFrame(list(normal_dict.items()), columns=['word', 'normal_frequency'])

#    count_df = pd.concat([blocked_df, normal_df], axis=1)
#    count_df.to_csv('word_frequencies.csv', index=False, encoding='utf-8-sig')
#    common_words.to_csv('common_words.csv', index=False, encoding='utf-8-sig')
# return common_words


def vectorizing_vectorizer(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform((df['text']))
    features_df = pd.DataFrame(vectors.todense(), columns=vectorizer.get_feature_names_out())
    return features_df


def find(x):
    if x == 1:
        print("Message is SPAM")
    else:
        print("Message is NOT SPAM")


def preprocess_files():
    if not os.path.exists(check_data_preprocessed):
        file_list = os.listdir(SEPARATED_DATA_PATH)

        for i in tqdm(range(len(file_list))):
            file_name = file_list[i]
            file_path = os.path.join(SEPARATED_DATA_PATH, file_name)
            df = replacing_commas(file_path)
            df = separating_colummns(df)
            df = grouping_by(df)
            df = duration_processing(df)
            df = total_chats_count(df)
            df = total_recipients_count(df)
            dfc = concatenate(df)
    with open(r"pickle.pkl", "rb") as f:
        dfc = pickle.load(f)

    dfc = pd.DataFrame(dfc)
    df = lemmatize(dfc)

    return df


def prepare_dfs_for_train(df):
    fraction_to_sample = 0.5
    df = df.sample(frac=fraction_to_sample, random_state=1)
    number_df = label_dummy_coding(drop_columns(df))
    numerical_df = numerical(number_df)
    normalized_df = normalize_numerical(numerical_df)
    tfidf_df = vectorizing_vectorizer(number_df)
    features_df = pd.concat([normalized_df, tfidf_df], axis=1, ignore_index=True)
    features_path = os.path.join(NORMALIZED_DATA_PATH, f"normalized_features.parquet")
    lable_df_path = os.path.join(NORMALIZED_DATA_PATH, f"lable_df.parquet")
    features_df.to_parquet("normalized_features.parquet", compression='gzip')
    label_df = number_df.filter(["id", "label"])
    label_df.to_parquet("label_df.parquet", compression='gzip')
    return features_df, label_df


from sklearn.decomposition import SparsePCA, TruncatedSVD
from scipy.sparse import csr_matrix


def perform_pca(X_sparse, n_components=100):
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X_sparse)
    return csr_matrix(X_reduced)


def train_files(data_for_models, lable_df):
    X_sparse = csr_matrix(data_for_models)
    X_sparse = perform_pca(X_sparse)
    X_train, X_test, y_train, y_test = train_test_split(X_sparse, lable_df.iloc[:, 0], test_size=0.15,
                                                        random_state=111)

    models_list = {'Logistic Regression': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(),
                   'Random Forest': RandomForestClassifier(), 'SVC': SVC()}
    pred_scores_word_vectors = {}
    results = {}
    from sklearn.model_selection import StratifiedKFold
    n_splits = 100
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)  # чи треба тут random state

    for name, classifier in tqdm(models_list.items()):
        accuracy_scores = []
        classification_reports = []
        confusion_matrices = []
        y_preds = []

        for train_idx, test_idx in tqdm(skf.split(X_sparse, label_df['label'])):
            X_train, X_test = X_sparse[train_idx], X_sparse[test_idx]
            y_train, y_test = label_df['label'].iloc[train_idx], label_df['label'].iloc[test_idx]

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            y_preds.append(y_pred)

            accuracy = accuracy_score(y_test, y_pred)
            pred_scores_word_vectors[name] = accuracy
            accuracy_scores.append(accuracy)

            classification_rep = classification_report(y_test, y_pred,
                                                       target_names=['class_0_normal', 'class_1_spammers'])
            #results[name]['Classification Report'] = classification_rep
            classification_reports.append(classification_rep)

            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices.append(cm)

        mean_y_pred = np.concatenate(y_preds)
        mean_accuracy = sum(accuracy_scores) / n_splits
        mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(mean_confusion_matrix, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        plt.title(f"Confusion Matrix - {name} - {mean_accuracy}")
        plt.savefig(f"Confusion Matrix - {name} - {mean_accuracy}.png")
        plt.close()

        mean_classification_report = classification_report(y_test, y_pred,
                                                           target_names=['class_0_normal', 'class_1_spammers'])

        report_file_path = f'classification_report_{name}.txt'

        # Open the file in write mode and write the classification report content
        with open(report_file_path, 'w') as report_file:
            report_file.write(mean_classification_report)

    best_clf_key = max(pred_scores_word_vectors, key=lambda k: pred_scores_word_vectors[k])
    best_clf_value = models_list.get(best_clf_key)
    explainer = shap.Explainer(best_clf_value, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    print(shap.summary_plot(shap_values, X_test))
    BEST_CLF = best_clf_value
    print(best_clf_key, best_clf_value)
    filename = f'{BEST_CLF}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(BEST_CLF, file)

    return results


def prepare_data_for_train(dataframe):
    features_df, label_df = prepare_dfs_for_train(dataframe)
    features_path = os.path.join(ROOT_PATH, f"normalized_features.parquet")
    table = pq.read_table("normalized_features.parquet")
    features_df = table.to_pandas()
    return features_df, label_df


if __name__ == '__main__':
    files_list = []
    dataframes = pd.DataFrame()
    check_raw_data_separated = os.path.join(SEPARATED_DATA_PATH, "blocked_0.parquet")
    check_data_preprocessed = os.path.join(ROOT_PATH, f"pickle.pkl")
    check_data_lemmatized = os.path.join(PROCESSED_DATA_PATH, "preprocessed_data_1.csv")
    check_data_normalized = os.path.join(ROOT_PATH, "lable_df.parquet")
    check_data_trained = os.path.join(PROCESSED_DATA_PATH, f"*.png")

    if not os.path.exists(check_raw_data_separated):
        file_names_list = ["blocked_user.csv", "regular_user.csv"]
        list_of_all = []
        for file_name in file_names_list:
            file_path = os.path.join(RAW_DATA_PATH, file_name)
            list_of_all.append(file_path)
            files_separation(file_path)

    if not os.path.exists(check_data_lemmatized):
        preprocess_files()

    if not os.path.exists(check_data_trained):
        if not os.path.exists(check_data_normalized):
            file_list = os.listdir(PROCESSED_DATA_PATH)
            print(len(file_list))
            dataframes = pd.DataFrame()
            for i in tqdm(range(len(file_list))):
                file_name = file_list[i]
                file_path = os.path.join(PROCESSED_DATA_PATH, file_name)
                df = pd.read_csv(file_path, encoding='utf=8=sig')
                df.columns = ["owner", "label", "total_recipients", "total_chats", "newest_message", "oldest_message",
                              "duration", 'chats_per_day', 'recipients_per_day', "text", "language"]
                dataframes = pd.concat([dataframes, df], axis=0, ignore_index=True)
                dataframes.columns = ["owner", "label", "total_recipients", "total_chats", "newest_message",
                                      "oldest_message", "duration", 'chats_per_day', 'recipients_per_day', "text",
                                      "language"]

            features_df, label_df = prepare_data_for_train(dataframes)
            trained_data = train_files(features_df, label_df)
        else:
            features_path = os.path.join(ROOT_PATH, "normalized_features.parquet")
            pf = fastparquet.ParquetFile("normalized_features.parquet")
            table = pq.read_table("normalized_features.parquet")
            features_df = table.to_pandas()
            # features_df = fastparquet.ParquetFile("normalized_features.parquet").to_pandas()
            label_df_path = os.path.join(ROOT_PATH, "label_df.parquet")
            label_df = fastparquet.ParquetFile("label_df.parquet").to_pandas()
            trained_data = train_files(features_df, label_df)

    # for name, classifier in tqdm(models_list.items()):
    #     classifier.fit(X_train, y_train)
    #     y_pred = classifier.predict(X_test)
    #
    #     accuracy = accuracy_score(y_test, y_pred)
    #     results[name] = {'Accuracy': accuracy}
    #     pred_scores_word_vectors[name] = accuracy
    #
    #     classification_rep = classification_report(y_test, y_pred, target_names=['class_0_normal', 'class_1_spammers'])
    #     results[name]['Classification Report'] = classification_rep
    #     report_file_path = f'classification_report_{classifier}.txt'
    #
    #     # Open the file in write mode and write the classification report content
    #     with open(report_file_path, 'w') as report_file:
    #         report_file.write(classification_rep)
    #
    #     # Generate and save confusion matrix plot
    #     cm = confusion_matrix(y_test, y_pred)
    #     f, ax = plt.subplots(figsize=(5, 5))
    #     sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
    #     plt.xlabel("y_pred")
    #     plt.ylabel("y_true")
    #     plt.title(f"Confusion Matrix - {name} - {accuracy}")
    #     plt.savefig(f'confusion_matrix_{name}.png')
    #     plt.close()
    #
    # best_clf_key = max(pred_scores_word_vectors, key=lambda k: pred_scores_word_vectors[k])
    # best_clf_value = models_list.get(best_clf_key)
    # # explainer = shap.Explainer(best_clf_value, X_train)
    # # shap_values = explainer.shap_values(X_test)
    # # shap.summary_plot(shap_values, X_test)
    # # print(shap.summary_plot(shap_values, X_test))
    # BEST_CLF = best_clf_value
    # print(best_clf_key, best_clf_value)
    # filename = f'{BEST_CLF}.pkl'
    # with open(filename, 'wb') as file:
    #     pickle.dump(BEST_CLF, file)
    # return results
