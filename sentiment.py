pip install TextBlob
import pandas as pd
data_set = pd.read_csv('Shoes_Data.csv')
data_set.head()
from textblob import TextBlob

# function to calculate polarity
def getPolarity(reviews):
    return TextBlob(reviews).sentiment.polarity

# function to calculate subjectivity 
def getSubjectivity(reviews):
    return TextBlob(reviews).sentiment.subjectivity

# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
fin_data = pd.DataFrame(data_set['reviews'])
# fin_data['Subjectivity'] = fin_data['Lemma'].apply(getSubjectivity) 
fin_data['Polarity'] = fin_data['reviews'].apply(getPolarity) 
fin_data['Analysis'] = fin_data['Polarity'].apply(analysis)
# print(fin_data.to_string())
review_data = pd.DataFrame(fin_data)
review_data
counts = fin_data.Analysis.value_counts()
counts
import matplotlib.pyplot as plt
%matplotlib inline

tb_counts= fin_data.Analysis.value_counts()
plt.figure(figsize=(10, 7))
plt.pie(tb_counts.values, labels = tb_counts.index, explode = (0, 0.25, 0.3), autopct='%1.1f%%', shadow=True)
plt.legend()
rat_count = data_set.rating.value_counts()
rat_count
round(data_set.rating.value_counts()*50,2).plot(kind='bar')

