# https://docs.google.com/document/d/17a19MP2KLCXvx66KW5Kql-1pFEA1DLNXL_BkULePQH0/edit?usp=sharing
# https://www.kaggle.com/datasets/atefehmirnaseri/cell-phone-price?select=CellPhone_train.csv

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

"""


battery_power: Общий заряд батареи, который можно сохранить (мАч)
blue: устройство оснащено функцией blutooth или нет, да/нет (1/0)
clock_speed: скорость выполнения инструкций микропроцессором
dual_sim: устройство имеет две SIM-карты одновременно или нет
fc: качество изображения фронтальной камеры в мегапикселях
four_g: устройство имеет сеть 4G или нет
int_memory: объем встроенной памяти в гигабайтах
m_dep: глубина устройства в сантиметрах
mobile_wt: вес устройства
n_cores: количество процессорных ядер
pc: качество основной камеры в мегапикселях
px_height: высота разрешения в пикселях, 
px_width: ширина разрешения в пикселях
ram: оперативная память в мегабайтах
sc_h: высота экрана устройства в сантиметрах
sc_w: ширина экрана устройства в сантиметрах
talk_time: максимальное время разговора, которое может поддерживать полностью заряженный аккумулятор устройства
three_g: устройство имеет Сеть 3G или нет
touch_screen: устройство имеет сенсорный экран или нет
wifi: устройство имеет Wi-Fi или нет
price_range: цена устройства по категориям
"""


def convert_to_minutes(row) -> int:
    duration = row['Duration']
    number_of_episodes = row['Episodes']
    time_parts = duration.split()
    if 'hr.' in duration:
        hours = int(time_parts[0])
        if 'min.' in duration:
            minutes = int(time_parts[2])
        else:
            minutes = 0
    else:
        hours = 0
        minutes = int(time_parts[0])
    return int(number_of_episodes * (hours * 60 + minutes))


# Preparing of the Pandas output and loading the dataset
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 250)
df = pd.read_csv("db_anime.csv")

# 1. How many rows and columns are there in the dataset?
rows, columns = df.shape
print(f'The dataset has {rows} rows and {columns} columns.')

# 2. Find out the data type of each feature
data_types = df.dtypes
print("Data types of each feature:\n", data_types)

# 3. Determine the data type of each feature from the point of view of analysis
# We categorize features as continuous, discrete, nominal, or ordinal.
# Continuous: Score, Popularity, Members, Duration
# Discrete: Rank, Episodes
# Nominal: Description, Synonyms, Japanese Title, English Title, Type, Status, Aired, Premiered, Broadcast, Producers, Licensors, Studios, Source, Genres, Rating
# Ordinal: Demographic (this could be considered ordinal if we assume a certain order in demographics)

# 4. Are there gaps in the data?
missing_data = df.isnull().sum()
print("Missing data in each column:\n", missing_data)

# Handling missing data
# For numeric columns, we can fill missing values with the median.
numeric_columns = ['Score', 'Popularity', 'Rank', 'Members', 'Episodes']
for column in numeric_columns:
    if df[column].isnull().sum() != 0:
        df[column].fillna(df[column].median(), inplace=True)
# For categorical columns, we can fill missing values with the mode.
categorical_columns = ['Description', 'Synonyms', 'Japanese', 'English', 'Type', 'Status', 'Aired', 'Premiered',
                       'Broadcast', 'Producers', 'Licensors', 'Studios', 'Source', 'Genres', 'Demographic', 'Duration',
                       'Rating']
for column in categorical_columns:
    if df[column].isnull().sum() != 0:
        df[column].fillna(df[column].mode()[0], inplace=True)

# Verify missing data handling
missing_data_after = df.isnull().sum()
print("Missing data after handling:\n", missing_data_after)

# Creating of new feature "Duration_in_minutes"
df['Duration_in_minutes'] = df.apply(convert_to_minutes, axis=1)
print(df['Duration_in_minutes'])


# 5. Are there outliers in the data?
# We'll use the IQR method to detect outliers

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    return outliers


# Detect outliers in numeric columns
numeric_columns.append('Duration_in_minutes')
outliers = {}
for column in numeric_columns:
    outliers[column] = detect_outliers(df, column)
    print(f'Outliers in {column}:\n', outliers[column])

# Removing outliers?
for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

# Calculate descriptive statistics for variables
descriptive_stats = df.describe()
print("Descriptive statistics:\n", descriptive_stats)

# 6. Interpret descriptive statistics for one numeric attribute and for one categorical one.
# Numeric attribute: Score
print("Descriptive statistics for 'Score':\n", descriptive_stats['Score'])

# Categorical attribute: Type
type_counts = df['Type'].value_counts()
print("Descriptive statistics for 'Type':\n", type_counts)

# 7. Build at least three graphs (moreover, you need to use at least two types of visualizations)
# Boxplot of Scores
plt.figure(figsize=(10, 5))
plt.boxplot(df['Score'], vert=False, patch_artist=True, medianprops={'linewidth': 3})
plt.title('Boxplot of Scores')
plt.xlabel('Score')
plt.grid(True)
plt.show()

# Bar plot of Type counts
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Type')
plt.title('Count of Anime Types')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

# Scatter plot of Score vs Popularity
plt.figure(figsize=(10, 5))
# Add a regression line
sns.regplot(data=df, x='Score', y='Popularity', scatter_kws={'alpha': 0.5})
plt.title('Score vs Popularity')
plt.xlabel('Score')
plt.ylabel('Popularity')
plt.show()

# 8. Build a correlation matrix for quantitative variables
correlation_matrix = df[numeric_columns].corr()
print("Correlation matrix:\n", correlation_matrix)
plt.figure(figsize=(12, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 9. Implement 3 of the proposed 5 hypotheses

# 9.1 Perform a z-test for mathematical expectation
# Hypothesis: The average score of an anime is 7.5
mean_score = df['Score'].mean()
std_score = df['Score'].std()
n = len(df['Score'])
z_score = (mean_score - 7.5) / (std_score / np.sqrt(n))
p_value = stats.norm.sf(abs(z_score)) * 2  # two-tailed test
print(f'Hypothesis: The average score of an anime is 7.5\n'
      f'    Z-test: z_score = {z_score}, p_value = {p_value}')

# 9.2 Take a t-test for mathematical expectation
# Hypothesis: The average popularity of an anime is different from 5000
t_stat, p_value = stats.ttest_1samp(df['Popularity'], 5000)
print(f'Hypothesis: The average popularity of an anime is different from 5000\n'
      f'    T-test: t_stat = {t_stat}, p_value = {p_value}')

# 9.3 Perform a test for the equality of mathematical expectations of two samples
# Hypothesis: The average score of TV series is different from movies
tv_scores = df[df['Type'] == 'TV']['Score']
movie_scores = df[df['Type'] == 'Movie']['Score']
t_stat, p_value = stats.ttest_ind(tv_scores, movie_scores)
print(f'Hypothesis: The average score of TV series is different from movies\n'
      f'    Test for equality of means: t_stat = {t_stat}, p_value = {p_value}')

# 10. Build a linear or logistic regression of at least 3 features
# Predicting Score based on Popularity, Members, and Duration_in_minutes
X = df[['Popularity', 'Members', 'Duration_in_minutes']]
y = df['Score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = LinearRegression()
model.fit(X_train, y_train)

# 11. Evaluate the quality of the model
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print(f'Mean Absolute Error: {mae}')
