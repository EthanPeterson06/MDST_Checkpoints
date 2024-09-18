import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('states_edu.csv')

# focusing on GRADE 4 Math

cleaned = df[df['AVG_MATH_4_SCORE'].notna()]
years = cleaned.groupby('YEAR').nunique()
print(len(years))

mi_data = cleaned[cleaned['STATE'] == 'MICHIGAN']

avg_score_mi = mi_data['AVG_MATH_4_SCORE'].mean()
print(avg_score_mi)

oh_data = cleaned[cleaned['STATE'] == 'MICHIGAN']

avg_score_oh = oh_data['AVG_MATH_4_SCORE'].mean()
print(avg_score_oh)

avg_2019 = df[df['YEAR'] == 2019]['AVG_MATH_4_SCORE'].dropna().mean()
print(avg_2019)

max_scores_per_state = cleaned.groupby('STATE')['AVG_MATH_4_SCORE'].max()
print(max_scores_per_state)

more_cleaned = df.dropna()
more_cleaned.loc[:, 'INSTRUCTION_EXPENDITURE_OUT_OF_TOTAL_REVENUE'] = more_cleaned['INSTRUCTION_EXPENDITURE'] / more_cleaned['TOTAL_REVENUE']
# I made this new column because I wanted to know how much each state values instruction. 

plt.scatter(more_cleaned['AVG_MATH_4_SCORE'], more_cleaned['TOTAL_REVENUE'], alpha=0.6)
plt.title('Total Revenue vs Avg Math 4 Score')
plt.xlabel('Total Revenue')
plt.ylabel('Avg Math 4 Score')
plt.show()

# I'm curious about how Revenue affects math scores in grade 4. I would assume that more money to the state means that more money would go toward education and better the scores, but to what point?
# The result of the plot shows a bell curve, with scores increasing until about 245, then decreasing after.

plt.scatter(more_cleaned['AVG_MATH_4_SCORE'], more_cleaned['INSTRUCTION_EXPENDITURE'], alpha=0.6)
plt.title('Instruction Expenditure vs Avg Math 4 Score')
plt.xlabel('Insturction Expenditure')
plt.ylabel('Avg Math 4 Score')
plt.show()

# I want to know how if insturtcion expenditure increases performance on math tests for grade 4. Does the rate of increase in performance ever get reduced?
# The result of the plot shows a bell curve, with the scores increasing until about 245, then decreasing after.

from sklearn.model_selection import train_test_split
X = more_cleaned[['INSTRUCTION_EXPENDITURE','TOTAL_REVENUE','AVG_READING_4_SCORE']].dropna()
y = more_cleaned.loc[X.index]['AVG_MATH_4_SCORE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print(model.intercept_)
print(model.coef_)

model.score(X_test, y_test)
np.mean(model.predict(X_test)-y_test)
np.mean(np.abs(model.predict(X_test)-y_test))
np.mean((model.predict(X_test)-y_test)**2)**0.5

col_name = 'INSTRUCTION_EXPENDITURE'

f = plt.figure(figsize=(12,6))
plt.scatter(X_train[col_name], y_train, color = "red")
plt.scatter(X_train[col_name], model.predict(X_train), color = "green")

plt.legend(['True Training','Predicted Training'])
plt.xlabel(col_name)
plt.ylabel('Math 4 score')
plt.title("Model Behavior On Training Set")
plt.show()

col_name = 'TOTAL_REVENUE'

f = plt.figure(figsize=(12,6))
plt.scatter(X_test[col_name], y_test, color = "blue")
plt.scatter(X_test[col_name], model.predict(X_test), color = "black")

plt.legend(['True testing','Predicted testing'])
plt.xlabel(col_name)
plt.ylabel('Reading 8 score')
plt.title("Model Behavior on Testing Set")
plt.show()



