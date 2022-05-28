from typing import List
import pandas as pd
import sqlite3

"""1 Belgium
1729 England
4769 France
7809 Germany
10257 Italy
13274 Netherlands
15722 Poland
17642 Portugal
19694 Scotland
21518 Spain
24558 Switzerland"""

COUNTRY = 21518

conn = sqlite3.connect("database.sqlite")

def get_team_ids(code):
    return pd.read_sql("""SELECT DISTINCT HT.team_long_name AS  home_team,
                                    HT.team_api_id AS team_id
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                WHERE Country.id = '{0}';""".format(code), conn).set_index('home_team').T.to_dict('list')


def team_matches(name):
    print(name)
    detailed_matches = pd.read_sql("""SELECT Match.id, 
                                            season, 
                                            stage, 
                                            date,
                                            HT.team_long_name AS  home_team,
                                            AT.team_long_name AS away_team,
                                            home_team_goal, 
                                            away_team_goal,
                                            goal,
                                            shoton,
                                            shotoff,
                                            foulcommit,
                                            card,
                                            cross,
                                            corner,
                                            possession                                    
                                    FROM Match
                                    JOIN Country on Country.id = Match.country_id
                                    JOIN League on League.id = Match.league_id
                                    LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                    LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                    WHERE (home_team='{0}'
                                    OR away_team='{0}')
                                    ORDER by date ASC;""".format(name), conn)
    return detailed_matches

def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def show(elem, indent=0):
    for child in elem:
        if not child:
            print('>' * indent, child.tag, child.text)
        else:
            print('>' * indent, child.tag)
        show(child, indent + 1)
