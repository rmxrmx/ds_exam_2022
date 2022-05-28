
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from goal_utils import *


team_ids = get_team_ids(COUNTRY)

print(team_ids)

bad_names = []
if COUNTRY == 21518:
    bad_names = ["Real Sporting de Gijón", "Málaga CF"]


for name in bad_names:
    del team_ids[name]


#print(season_goals("Real Sporting de Gijón"))
# fig, ax = plt.subplots()

# goals = pd.DataFrame(columns=["team", "season", "goals", "header", "shot", "freekick", "tap_in", "first30", "middle30", "last30", "unavailable"])
# for key in team_ids:
#     df = season_goals(key)
    
#     if len(df) == 8:
#         goals = pd.concat([goals, df])


# for name, group in goals.groupby("team"):
#     group.plot(x='season', y='unavailable', ax=ax, label=name)

# plt.show()

# print(goals)

matches =  team_matches("FC Barcelona")

print(matches)