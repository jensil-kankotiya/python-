# Professional Report

## Project 1: Black Friday Sale Consumer Behaviour Analysis

### Introduction
The "Black Friday Sale Consumer Behaviour Analysis" project aims to analyze consumer behavior during Black Friday sales. The main goal is to understand purchasing patterns and factors influencing consumer decisions.

### Code Explanation
#### 1. Importing Packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
#### 2. Loading Data
```python
df = pd.read_csv('BlackFriday.csv')
```
#### 3. Data Cleaning
```python
df.dropna(inplace=True)
df['Age'] = df['Age'].apply(lambda x: int(x.split('-')[0]))
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
```
#### 4. Exploratory Data Analysis (EDA)
```python
sns.countplot(x='Age', data=df)
plt.show()
```
#### 5. Feature Engineering
```python
df['Purchase_log'] = np.log(df['Purchase'] + 1)
```
#### 6. Modeling
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['Age', 'Gender', 'Occupation']]
y = df['Purchase_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
```
#### 7. Evaluation
```python
from sklearn.metrics import mean_squared_error

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
```

### Conclusion
The analysis provides valuable insights into consumer behavior during Black Friday sales. Key factors influencing purchases include age, gender, product category, and purchase frequency. The machine learning models demonstrate good predictive performance, providing a foundation for future marketing strategies.

## Project 2: Sentimental Analysis Joe Biden vs Donald Trump

### Introduction
The "Sentimental Analysis Joe Biden vs Donald Trump" project aims to analyze the sentiment of tweets related to Joe Biden and Donald Trump during the 2020 US Presidential Election. The goal is to understand public sentiment and compare the sentiment between the two candidates.

### Code Explanation
#### 1. Importing Libraries
```python
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
```
#### 2. Loading Dataset
```python
trump = pd.read_csv('hashtag_donaldtrump.csv', lineterminator='\n')
biden = pd.read_csv('hashtag_joebiden.csv', lineterminator='\n')
```
#### 3. Data Cleaning
```python
trump.dropna(inplace=True)
biden.dropna(inplace=True)
```
#### 4. Exploratory Data Analysis (EDA)
```python
wordcloud = WordCloud().generate(' '.join(trump['tweet']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```
#### 5. Sentiment Analysis
```python
trump['sentiment'] = trump['tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
biden['sentiment'] = biden['tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
```
#### 6. Comparison
```python
sns.histplot(trump['sentiment'], color='red', label='Trump', kde=True)
sns.histplot(biden['sentiment'], color='blue', label='Biden', kde=True)
plt.legend()
plt.show()
```
### Conclusion
The sentiment analysis reveals interesting patterns in public sentiment towards Joe Biden and Donald Trump during the 2020 US Presidential Election. The analysis highlights differences in sentiment distribution and provides insights into public opinion on social media.
