import pandas as pd
import matplotlib.pyplot as plt

# Python Challenge Part 1
# Read the dataset into a Pandas DataFrame and handle any missing or inconsistent data.
Insta_City = pd.read_csv(r"C:\PycharmProjects\FacebookInstagramPerformance\CSV_files\Insta_City.csv")
Insta_Overview = pd.read_csv(r"C:\PycharmProjects\FacebookInstagramPerformance\CSV_files\Insta_Overview.csv")
Insta_Post = pd.read_csv(r"C:\PycharmProjects\FacebookInstagramPerformance\CSV_files\Insta_Post.csv")
Insta_Age = pd.read_csv(r"C:\PycharmProjects\FacebookInstagramPerformance\CSV_files\Insta_Age.csv")
FB_Overview = pd.read_csv(r"C:\PycharmProjects\FacebookInstagramPerformance\CSV_files\Facebook_Overview.csv")
FB_Post = pd.read_csv(r"C:\Users\danie\PycharmProjects\FacebookInstagramPerformance\CSV_files\Facebook_Post.csv")

# Counts the Null Rows
def null_check():
    print(Insta_City.isna().sum())
    print(Insta_Overview.isna().sum())
    print(Insta_Post.isna().sum())
    print(Insta_Age.isna().sum())
    print(FB_Overview.isna().sum())
    print(FB_Post.isna().sum())
# Counts Total Rows
def row_count():
    print(Insta_City.count())
    print(Insta_Overview.count())
    print(Insta_Post.count())
    print(Insta_Age.count())
    print(FB_Overview.count())
    print(FB_Post.count())
# Information about each dataset
def data_info():
    print(Insta_City.info())
    print(Insta_Overview.info())
    print(Insta_Post.info())
    print(Insta_Age.info())
    print(FB_Overview.info())
    print(FB_Post.info())
# checks for duplicate rows
def check_unique():
    print(Insta_City[Insta_City.duplicated()])
    print(Insta_Overview[Insta_Overview.duplicated()])
    print(Insta_Post[Insta_Post.duplicated()])
    print(Insta_Age[Insta_Age.duplicated()])
    print(FB_Overview[FB_Overview.duplicated()])
    print(FB_Post[FB_Post.duplicated()])

# Running these functions we can see we have multiple NULL columns and some NULL values
# We can also see we have no duplicate rows
# Now we remove Null columns
Insta_Overview = Insta_Overview.drop("New followers", axis='columns')
FB_Overview = FB_Overview.drop("% of reach from organic", axis='columns')
# Remove Null Rows
Insta_Overview = Insta_Overview.dropna(how='any',axis=0)
FB_Overview = FB_Overview.dropna(how='any',axis=0)
FB_Post= FB_Post.dropna(how='any',axis=0)

# Python Challenge Part 2
# Calculate the average engagement rate for Instagram posts and
# identify the top-performing post based on engagement (likes, comments, shares).

# Sum of engagement divided by amount of posts
sum_engagement = sum(Insta_Overview['Engagement'])
sum_posts = len(Insta_Post)
aer = (sum_engagement/sum_posts)
aer = str(aer)
print("The average engagement rate per post is " + aer)

# Top performing posts based on the metrics stated above
toplikes = Insta_Post["Like count"].idxmax()
print(Insta_Post['Media caption'].iloc[toplikes])

topcomments = Insta_Post["Comments count"].idxmax()
print(Insta_Post['Media caption'].iloc[topcomments])

topshares = Insta_Post["Shares"].idxmax()
print(Insta_Post['Media caption'].iloc[topshares])

# Python Challenge 3
# Create a simple line chart showing post engagement trends over time

FB_Overview.plot.line(x='Date', y='Page post engagements')
plt.xlabel("Date")
plt.ylabel("Engagement")
plt.title("Engagement Over Time")
plt.show()

# Python Challenge 4 Bonus Task
# Write a function that predicts whether a post will perform well based on previous engagement data
# I've filtered the data to only start from the first of January. It's an arbitrary date but
# as there's been consistant and inconsistant uploading to instagram the model doesn't work
# well over a long period of time. With a shorter time frame it's also more accurate.

Insta_Overview["Date"] = pd.to_datetime(Insta_Overview["Date"], dayfirst=True)
filtered_df = Insta_Overview.loc[(Insta_Overview['Date'] >= '01/01/2025')]
data_t = list(range(len(filtered_df['Engagement'])))
data_y = filtered_df['Engagement']

def holt_alg(h, y_last, y_pred, T_pred, alpha, beta):
    pred_y_new = alpha * y_last + (1-alpha) * (y_pred + T_pred * h)
    pred_T_new = beta * (pred_y_new - y_pred)/h + (1-beta)*T_pred
    return (pred_y_new, pred_T_new)

def smoothing(t, y, alpha, beta):
    # initialization using the first two observations
    pred_y = y[1]
    pred_T = (y[1] - y[0])/(t[1]-t[0])
    y_hat = [y[0], y[1]]
    # next unit time point
    t.append(t[-1]+1)
    for i in range(2, len(t)):
        h = t[i] - t[i-1]
        pred_y, pred_T = holt_alg(h, y[i-1], pred_y, pred_T, alpha, beta)
        y_hat.append(pred_y)
    return y_hat

plt.plot(data_t, data_y, '-', label = "Actual Engagement")

pred_y = smoothing(data_t, data_y, alpha=.8, beta=.5)
plt.plot(data_t[:len(pred_y)], pred_y, 'r-.', label = "Predicted Engagement")
plt.xlabel("Date")
plt.ylabel("Engagement")
plt.title("A graph to show actual and predicted engagement over time")
plt.legend()
plt.show()
