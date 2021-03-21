import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
students_df = pd.read_csv("../../../data/Students_Performance.csv")
# df_count_ethnicities = students_df.groupby(['gender', "race.ethnicity"]).mean().reset_index()
# print((df_count_ethnicities))


# colored_by = "test.preparation.course"
# split_by = 'parental.level.of.education'
#
#
#
# for level in students_df[split_by].unique():
#     df = students_df.loc[students_df[split_by] == level].groupby([colored_by]).size().reset_index(name='Count')
#     d=(px.pie(df, values='Count', names = colored_by, title = level))
#     fig = go.Figure(
#         data=d,
#         layout_title_text="A Figure Displayed with fig.show()"
#     )
#     fig.show()

students_df["gender.cat"] = pd.Categorical(students_df["gender"]).codes

fig = make_subplots(rows=1, cols=2, start_cell="bottom-left")

fig.add_traces([go.Scatter(x=students_df["math.score"], y=students_df["reading.score"], mode="markers",
                           marker = dict(color = students_df["gender.cat"], colorscale="Bluered"), showlegend = False),
                go.Scatter(x=students_df["math.score"], y=students_df["science.score"], mode="markers",
                           marker = dict(color = students_df["gender.cat"], colorscale="Bluered"), showlegend = False)],
               rows=[1,1], cols=[1,2])
fig.add_trace(go.Scatter(x = [None], y = [None], mode = 'markers',
                        marker = dict(color="Blue"), legendgroup = "female", name = "female"), row = 1, col =1)
fig.add_trace(go.Scatter(x = [None], y = [None], mode = 'markers',
                        marker = dict(color="Red"), legendgroup = "male", name = "male"), row = 1, col =1)
fig.update_xaxes(title_text="Reading Score", row=1, col=1)
fig.update_xaxes(title_text="Science Score", row=1, col=2)
fig.update_yaxes(title_text="Math Score")
fig.show()


