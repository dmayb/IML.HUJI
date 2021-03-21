import pandas as pd
import numpy as np

# students_df = pd.DataFrame(np.array([np.random.choice(["Zohar", "Shelly", "Omer", "Avi"],50), np.random.choice(["Linearit", "Intro", "Infi", "Probabilistic"], 50), np.random.randint(80, 101, 50)]).transpose(), columns=['Name','Course','Grade'])
# students_df["Grade"] = students_df["Grade"].astype(int)
#
# print("\n\nStudents df")
# print(students_df.head())
#
# print("\n\nCalculate average by student and by course")
# print(students_df.groupby(['Name', 'Course']).mean().reset_index())
# print("\n\ngroupby")
# print(students_df.groupby(['Name', 'Course']).mean().head())


def create_flight_df(cities_poss, nrows = 100):

    flights = pd.DataFrame(np.random.ch)