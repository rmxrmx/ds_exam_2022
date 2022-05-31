import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import numpy as np
from scipy.stats import ttest_ind

events = pd.read_csv("events.csv")
events = events[events["time"] <= 90]
match_data = pd.read_csv("ginf.csv")


shots = events[events["event_type"] == 1][["shot_place", "location", "bodypart", "assist_method", "situation", "fast_break", "is_goal"]]

print("Size of initial data: ", len(events))
print("Non-goal %: ", 100 - 100 * sum(shots["is_goal"]) / len(shots))

def balance_dataset(shots):
    goals = sum(shots["is_goal"])
    print("# of shots that need to be added:", goals)

    shots = pd.concat([shots[shots["is_goal"] == 1], shots[shots["is_goal"] == 0].head(goals)])

    print("Portion of goals of all shots:", sum(shots['is_goal']) / len(shots['is_goal']))

    return shots


def x_y_data(data):
    X = data.drop("is_goal", axis=1)
    X = pd.get_dummies(X, columns=["shot_place", "location", "bodypart", "assist_method", "situation"])
    Y = data["is_goal"]
    
    return X, Y

# this was not used because the dataset represents how skewed the data is in real-life.
#shots = balance_dataset(shots)

train, test = train_test_split(shots)

trainX, trainY = x_y_data(train)
testX, testY = x_y_data(test)

model = LogisticRegression(max_iter=500)
model.fit(trainX, trainY)

print("F1_micro on train: ", cross_validate(model, trainX, trainY, scoring='f1_micro', cv=5, return_estimator=True)['test_score'].mean())

y_hat = model.predict(testX)

print(f"Accuracy on test data: {accuracy_score(testY, y_hat)}")
print('The classifier obtains an ROC-AUC of {}'.format(roc_auc_score(testY, model.predict_proba(testX)[:, 1])))

print(confusion_matrix(testY, y_hat))

########

shots, Y = x_y_data(shots)
shots["xG"] = model.predict_proba(shots)[:, 1]
shots["is_goal"] = Y

xG = shots["xG"]

heights,bins = np.histogram(xG,bins=100)

# Normalize
heights = heights/float(sum(heights))
binMids=bins[:-1]+np.diff(bins)/2.

events = events.join(xG)
events["xG"] = events.fillna(0)["xG"]

no_of_games = len(events.id_odsp.unique())

print("xG in total: ", sum(events["xG"]))
print("Actual G in total: ", sum(events["is_goal"]))

xG_per_min = []
xG_error = []
events_per_min = []
x = np.array(range(1, 91))
for i in x:
    xG_per_min.append(events[(events["time"] == i) & (events["event_type"] == 1)]["xG"].mean())
    xG_error.append(events[(events["time"] == i) & (events["event_type"] == 1)]["xG"].std())
    events_per_min.append(len(events[(events["time"] == i)])) 


trend_model = LinearRegression()

x = x.reshape(-1, 1)
trend_model.fit(x, xG_per_min)

trend = trend_model.predict(x)

detrended_xg = xG_per_min - trend

print("DETRENDED XG MEAN: ", np.mean(detrended_xg))

# uncomment whatever plot you want to produce

# plt.plot(x, xG_per_min, label='mean xG each minute')
# plt.plot(x, detrended_xg, label='mean xG each minute (detrended)')
# plt.plot(x, events_per_min, label='events each minute')
# plt.plot(binMids,heights)
# plt.plot(x, trend, linestyle='--', color="purple", label='trend')

plt.grid(True)
plt.legend(loc='best')
plt.xlabel("xG")
plt.ylabel("density")
plt.xticks([0, 10, 20, 30, 40, 45, 50, 60, 70, 80, 90])
plt.title("Distribution of xG values")

# plt.show() 

def xg_before_after(events, event):
    # take only events that are: in the same game, from the same team, are a shot
    # NOTE: the second condition should be changed to != if the effects on the opposition are wanted
    match = events[(events["id_odsp"] == event["id_odsp"]) & (events["event_team"] == event["event_team"]) & (events["event_type"] == 1)]

    before = match[match["sort_order"] < event["sort_order"]]
    after = match[match["sort_order"] > event["sort_order"]]


    # Gives average xG per EVENT, not per TIME
    before_xg = before["xG"].tolist()
    after_xg = after["xG"].tolist()

    before_time = before["time"].tolist()
    after_time = after["time"].tolist()

    # detrend each individual event
    for i in range(len(before_xg)):
        before_xg[i] -= trend[before_time[i] - 1]

    for i in range(len(after_xg)):
        after_xg[i] -= trend[after_time[i] - 1]

    return before_xg, after_xg

# this should be changed depending on which event should be evaluated

# high_impact_event = events[(events["event_type2"] == 14) & (events["time"] < 90)] # sending off
# high_impact_event = events[(events["event_type"] == 1) & (events["location"] == 14) & (events["time"] < 90)] # getting a penalty
# high_impact_event = events[(events["event_type"] == 1) & (events["location"] == 14) & (events["is_goal"] == 1) & (events["time"] < 90)] # scoring a penalty
# high_impact_event = events[(events["event_type"] == 1) & (events["location"] == 14) & (events["is_goal"] == 0) & (events["time"] < 90)] # missing a penalty
high_impact_event = events[(events["event_type"] == 7) & (events["time"] < 46)] # substitution in first half


avg_b = []
avg_a = []
print("Number of high-impact events: ", len(high_impact_event))
for i in range(len(high_impact_event)):
    if i % 100 == 0:
        print("Iteration #", i)

    event = high_impact_event.iloc[i]

    b, a = xg_before_after(events, event)

    avg_b += b
    avg_a += a

print("Average xG before and after the high-impact event:", np.mean(avg_b), np.mean(avg_a))
print("STDs: ", np.std(avg_b), np.std(avg_a))
print("TTest: ", ttest_ind(avg_b, avg_a))
