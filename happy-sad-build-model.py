import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.utils import shuffle

seed = 333

df_sad = pd.read_json(r"reddit-sad.json")
df_sad = shuffle(df_sad, random_state=seed)
df_sad = df_sad.iloc[:10000]
df_sad = df_sad[['body']]
df_sad = df_sad.replace('\n', ' ', regex=True)
df_sad['class'] = 0


print("sad text dataframe: ")
print(df_sad.head())
print("\n")
print(df_sad.iloc[100, 0])
print("length: ", len(df_sad))


#get sentiment scores:
#if you don't have it, download vader lexicon:
#nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
df_sad['sentiment_scores_body'] = df_sad['body'].apply(lambda text: sia.polarity_scores(text))
df_sad['compound'] = df_sad['sentiment_scores_body'].apply(lambda score_dict: score_dict['compound'])

average_sad_sentiment = df_sad['compound'].mean()
print("average sad sentiment score: ", average_sad_sentiment)

#get subjectivity via textblob
df_sad['subj_scores'] = df_sad['body'].apply(lambda text: TextBlob(text).sentiment.subjectivity)
average_sad_subjectivity = df_sad['subj_scores'].mean()
print("average sad subjectivity score: ", average_sad_subjectivity)


df_happy = pd.read_csv(r"hm_train.csv")
df_happy = shuffle(df_happy, random_state=seed)
df_happy = df_happy.iloc[:10000]
df_happy = df_happy[['cleaned_hm']]
df_happy = df_happy.rename(columns={'cleaned_hm': 'body'})
df_happy['class'] = 1

print("happy text dataframe:")
print(df_happy.head())
print("\n")
print(df_happy.iloc[100, 0])
print("length: ", len(df_happy))

df_happy['sentiment_scores_body'] = df_happy['body'].apply(lambda text: sia.polarity_scores(text))
df_happy['compound'] = df_happy['sentiment_scores_body'].apply(lambda score_dict: score_dict['compound'])

average_happy_sentiment = df_happy['compound'].mean()
print("average happy sentiment score: ", average_happy_sentiment)

#get subjectivity via textblob
df_happy['subj_scores'] = df_happy['body'].apply(lambda text: TextBlob(text).sentiment.subjectivity)
average_happy_subjectivity = df_happy['subj_scores'].mean()
print("average happy subjectivity score: ", average_happy_subjectivity)

df = pd.concat([df_sad, df_happy], axis = 0)
print(df.head())

df = df.dropna()
df.to_csv("data.csv", index=False)