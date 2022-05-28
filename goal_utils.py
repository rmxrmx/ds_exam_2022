from utils import get_team_ids, team_matches
import xml.etree.ElementTree as ET
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd
from utils import COUNTRY

team_ids = get_team_ids(COUNTRY)

def get_goals(team):
    df = pd.DataFrame(columns=["date", "season", "elapsed", "type"])
    matches = team_matches(team)
    team_id = team_ids[team][0]

    for index, row in matches.iterrows():
        goal_info = ET.fromstring(row["goal"])

        for child in goal_info:
            if child.find("team") is not None and child.find("team").text == str(team_id):
                type_el = child.find("subtype")
                if type_el is not None:
                    type = type_el.text
                else:
                    type = "not available"


                season = (datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S") + relativedelta(months=-7)).year

                if type not in ["saved", "missed", "saved_back_into_play"]:
                    df.loc[len(df.index)] = [row["date"], season, int(child.find("elapsed").text), type]
                    
    return df

def season_goals(team):
    goals_df = get_goals(team)
    season_df = pd.DataFrame(columns=["team", "season", "goals", "header", "shot", "freekick", "tap_in", "first30", "middle30", "last30", "unavailable"])

    seasons = goals_df.season.unique()
    for i in seasons:
        goals = len(goals_df[goals_df['season'] == i])
        header = len(goals_df[(goals_df['season'] == i) & (goals_df['type'] == "header")]) * 100 / goals
        freekick = len(goals_df[(goals_df['season'] == i) & (goals_df['type'] == "direct_freekick")]) * 100 / goals
        tap_in = len(goals_df[(goals_df['season'] == i) & (goals_df['type'] == "tap_in")]) * 100 / goals
        shot = len(goals_df[(goals_df['season'] == i) & (goals_df['type'] != "tap_in") & (goals_df['type'] != "direct_freekick") & (goals_df['type'] != "header")]) * 100 / goals
        first30 = len(goals_df[(goals_df['season'] == i) & (goals_df['elapsed'] <= 30)]) * 100 / goals
        middle30 = len(goals_df[(goals_df['season'] == i) & (goals_df['elapsed'] > 30) & (goals_df['elapsed'] <= 60)]) * 100 / goals
        last30 = len(goals_df[(goals_df['season'] == i) & (goals_df['elapsed'] > 60)]) * 100 / goals
        
        unavailable = len(goals_df[(goals_df['season'] == i) & (goals_df['type'] == "not available")]) * 100 / goals

        season_df.loc[len(season_df.index)] = [team, i, goals, header, shot, freekick, tap_in, first30, middle30, last30, unavailable]
    return season_df