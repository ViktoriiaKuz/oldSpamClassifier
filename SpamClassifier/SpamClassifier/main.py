"""this code accepts the raw files of spammers and normal users messages and other parameters
(such as number of recipients/first and last message sent, number of chats of the specified type,
 and returns trained model which can classify new users according to their messages and parameters it trains different models,
 indicates the level of accuracy and save the one which have the best output parameters. The code is built in the way to reduce
 processing time and "save to files/pickles", in order to restart the process from the middle
 (in case if part of job is already done)"""
import os
import pickle
import sys
import warnings
from datetime import datetime
from io import StringIO

import dask.dataframe as dd
import langdetect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidLength
from pandas.errors import SettingWithCopyWarning
from scipy.sparse import hstack, csr_matrix, load_npz, save_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from settings import NORMALIZED_DATA_PATH
from settings import PROCESSED_DATA_PATH
from settings import RAW_DATA_PATH
from settings import ROOT_PATH
from settings import SEPARATED_DATA_PATH

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
header_text = ['owner', 'total_recipients', 'total_chats', 'newest_message', 'oldest_message', 'label', 'text',
               'duration',
               'chats_per_day', 'recipients_per_day', 'language']

header = ['owner', 'total_recipients', 'total_chats', 'newest_message', 'oldest_message', 'label', 'duration',
          'chats_per_day', 'recipients_per_day', 'text', 'language']
columns = ["0", "owner", "label", "total_recipients", "total_chats", "newest_message", "oldest_message",
           "duration", 'chats_per_day', 'recipients_per_day', "text", "language"]


def files_separation(file_paths):
    marker = os.path.basename(file_paths).split('_')[0]
    df = pd.read_csv(file_paths, encoding="utf-8-sig")
    list_df = [df[i:i + ROWS_TO_SEPARATE] for i in range(0, len(df), ROWS_TO_SEPARATE)]
    for i, df in tqdm(enumerate(list_df)):
        df[LABEL_KEY] = marker
        df.to_parquet(os.path.join(SEPARATED_DATA_PATH, f"{marker}_{i}.parquet"), index=False, compression='gzip')


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
    processed_df.to_csv(os.path.join(PROCESSED_DATA_PATH, f'preprocessed_data_{i}.csv'), encoding='utf-8-sig')






# def text_df_count(df):
#    text_spam = df[df['label'] == 'blocked']['text']
#    text_normal = df[df['label'] == 'regular']['text']
#
#    blocked_dict = {}
#    normal_dict = {}
#
#    for text in text_spam:
#        words = text.split()
#        for word in words:
#            blocked_dict[word] = blocked_dict.get(word, 0) + 1
#
#    for text in text_normal:
#        words = text.split()
#        for word in words:
#            normal_dict[word] = normal_dict.get(word, 0) + 1
#
#    common_words = set(blocked_dict.keys()) & set(normal_dict.keys())
#    common_words = pd.DataFrame(common_words)
#
#    blocked_dict = {word: freq for word, freq in blocked_dict.items() if word not in common_words}
#    normal_dict = {word: freq for word, freq in normal_dict.items() if word not in common_words}
#
#    blocked_df = pd.DataFrame(list(blocked_dict.items()), columns=['word', 'spam_frequency'])
#    normal_df = pd.DataFrame(list(normal_dict.items()), columns=['word', 'normal_frequency'])
#
#    count_df = pd.concat([blocked_df, normal_df], axis=1)
#    count_df.to_csv('word_frequencies.csv', index=False, encoding='utf-8-sig')
#    common_words.to_csv('common_words.csv', index=False, encoding='utf-8-sig')




def preprocess_files():
    print("002")
    if not os.path.exists(check_data_preprocessed):
        print("003")
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
    print("020", sys.getsizeof(df))  # df size 790275614
    drop_in_number = ['newest_message', 'oldest_message', 'duration', 'language', 'owner', 'label', 'text']
    df = df.sample(frac=0.5, random_state=1)
    number_df = df.drop(columns=[c for c in drop_in_number if c in df])
    number_feature_names = number_df.columns
    number_sparse = csr_matrix(MinMaxScaler().fit_transform(number_df))
    print("021", sys.getsizeof(number_sparse)) #56

    vac = TfidfVectorizer(stop_words='english')
    text_series = df['text']
    tfidf_sparse = vac.fit_transform(text_series)

    feature_sparse = hstack([number_sparse, tfidf_sparse])
    feature_names = list(number_feature_names) + list(vac.get_feature_names_out())

    print(feature_sparse.shape)

    print("022", sys.getsizeof(feature_sparse)) #56



    features_path = os.path.join(NORMALIZED_DATA_PATH, f"feature_sparse.npz")
    lable_path = os.path.join(NORMALIZED_DATA_PATH, f"lable_sparse.npz")
    feature_names_path = os.path.join(NORMALIZED_DATA_PATH, f"feature_names.npz")

    label_df = df.filter(["label"])
    label_sparse = csr_matrix(label_df)

    save_npz(features_path, feature_sparse)
    save_npz(lable_path, label_sparse)
    np.savez_compressed(feature_names_path, feature_names)

    exit(0)
    return feature_sparse, label_sparse, feature_names



# def perform_pca(X_sparse, n_components=100):
#     svd = TruncatedSVD(n_components=n_components)
#     X_reduced = svd.fit_transform(X_sparse)
#     return csr_matrix(X_reduced)


def train_files(feature_sparse, label_sparse, feature_names):
    print("009")
    models_list = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()  # ,
        # 'SVC': SVC()
    }

    pred_scores_word_vectors = {}
    n_splits = 5
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=10, random_state=42)

    print(feature_sparse.shape) #12131,167242
    feature_dense = feature_sparse.toarray()
    label_dense = label_sparse.toarray()
    #print(feature_names.shape) #167242

    #feature_sparse = vstack([feature_names, feature_sparse])

    for name, classifier in tqdm(models_list.items()):
        confusion_matrices = []
        accuracy_list = []
        classification_reports_fds = []
        mse_list = []


        #   data_for_models = data_for_models.to_pandas()
        print("019", sys.getsizeof(feature_dense))
        # print("048", label_df['label'])
        #data_for_models.reset_index(drop=True, inplace=True)

        for repeat_idx, (train_idx, test_idx) in enumerate(skf.split(feature_dense, label_dense)):
            X_train, X_test = feature_dense[train_idx], feature_dense[test_idx]
            y_train, y_test = label_dense[train_idx], label_dense[test_idx]

            # Now you can use the 'fold' variable to keep track of the current fold number
            print(f"Repeat {repeat_idx + 1}, Fold {repeat_idx % n_splits + 1}:")

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            #mse = mean_squared_error(y_test, y_pred)


            pred_scores_word_vectors[name] = accuracy
            accuracy_list.append(accuracy)
            #mse_list.append(mse)

            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices.append(cm)

            classification_rep = classification_report(
                y_test, y_pred, target_names=['class_0_normal', 'class_1_spammers'], output_dict=True
            )
            classification_rep_df = pd.DataFrame(classification_rep)
            classification_reports_fds.append(classification_rep_df)
            print("f1", accuracy)

        # Calculate mean confusion matrix
        print("010")
        mean_confusion_matrix = np.mean(confusion_matrices, axis=0)


        mean_accuracy = np.mean(accuracy_list)
        min_accuracy = np.min(mean_accuracy)
        #max_accuracy = np.max(mean_accuracy)
        std = np.sqrt(np.std(accuracy_list))

        # Plot and save the mean confusion matrix
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(mean_confusion_matrix, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        plt.title(f"Confusion Matrix - {name} - {min_accuracy}")
        plt.savefig(f"Mean Confusion Matrix - {name} - {min_accuracy}.png")
        plt.close()
        mean_classification_report = pd.concat(classification_reports_fds).groupby(level=0).mean()

        mean_classification_report = mean_classification_report.rename(
            columns={
                'precision': 'Mean Precision',
                'recall': 'Mean Recall',
                'f1-score': 'Mean F1-Score',
                'support': 'Mean Support'
            }
        )
        print("std", std)
        #mean_classification_report = pd.DataFrame(mean_classification_report)

        mean_classification_report.to_csv(f"classification_report + {name} + {mean_accuracy} +- {std} .csv",
                                          encoding='utf-8-sig')

    best_clf_key = max(pred_scores_word_vectors, key=lambda k: pred_scores_word_vectors[k])
    best_clf_value = models_list.get(best_clf_key)

    # model = best_clf_value
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # explainer = shap.Explainer(model, X_train)
    # instance_idx = 0
    # shap_values = explainer(X_test[instance_idx])
    # for i, shap_value in enumerate(shap_values.values):
    #     feature_name = feature_names[i]
    #     print(f"{feature_name}: {shap_value}")

    BEST_CLF = best_clf_value
    print(best_clf_key, best_clf_value)
    filename = f'{BEST_CLF}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(BEST_CLF, file)





if __name__ == '__main__':
    files_list = []
    dataframes = pd.DataFrame()
    check_raw_data_separated = os.path.join(SEPARATED_DATA_PATH, "blocked_0.parquet")
    check_data_preprocessed = os.path.join(ROOT_PATH, f"pickle.pkl")
    check_data_lemmatized = os.path.join(PROCESSED_DATA_PATH, "preprocessed_data_1.csv")
    check_data_normalized = os.path.join(NORMALIZED_DATA_PATH, "lable_sparse.npz")
    check_data_trained = os.path.join(PROCESSED_DATA_PATH, f"*.png")

    if not os.path.exists(check_raw_data_separated):
        print("004")
        file_names_list = ["blocked_user.csv", "regular_user.csv"]
        list_of_all = []
        for file_name in file_names_list:
            file_path = os.path.join(RAW_DATA_PATH, file_name)
            list_of_all.append(file_path)
            files_separation(file_path)

    if not os.path.exists(check_data_lemmatized):
        print("005")
        preprocess_files()

    if not os.path.exists(check_data_trained):
        print("006")
        if not os.path.exists(check_data_normalized):
            print("007")
            file_list = os.listdir(PROCESSED_DATA_PATH)
            df = pd.DataFrame()
            data = []

            for i in tqdm(range(len(file_list))):
                file_name = file_list[i]
                file_path = os.path.join(PROCESSED_DATA_PATH, file_name)
                with open(file_path, 'r', encoding='utf-8-sig') as file:
                    for line in file:
                        try:
                            data.append(line.strip())
                        except UnicodeDecodeError:
                            continue

                df = pd.read_csv(StringIO('\n'.join(data)), encoding='utf-8-sig')
                # df.columns = columns
                dataframes = pd.concat([dataframes, df], axis=0, ignore_index=True)
                dataframes.columns = columns

            features_df, label_df = prepare_dfs_for_train(dataframes)
            trained_data = train_files(features_df, label_df)
        else:
            print("008")
            features_path = os.path.join(NORMALIZED_DATA_PATH, "feature_sparse.npz")
            feature_sparse = load_npz(features_path)
            lable_path = os.path.join(NORMALIZED_DATA_PATH, "lable_sparse.npz")
            label_sparse = load_npz(lable_path)
            feature_names_path = os.path.join(NORMALIZED_DATA_PATH,  "feature_names.npz")
            feature_names = np.load(feature_names_path)

            stored_objects = feature_names.files

            # Iterate through the stored objects and print their names
            for obj_name in stored_objects:
                print(f"Stored object name: {obj_name}")
            if 'arr_0' in feature_names:
                feature_names = feature_names['arr_0']
                print(feature_names.shape) #167242,
            print("787")
            trained_data = train_files(feature_sparse, label_sparse, feature_names)
