import pandas as pd

df = pd.read_csv(r"data.csv")
df1 = df[(df['class'] == 1)]
df2 = df[(df['class'] == 0)]


ax = df1.plot.scatter(x="subj_scores", y="compound", color="DarkRed", label="Class: Happy")

print(df2.plot.scatter(x="subj_scores", y="compound", color="DarkGreen", label="Class: Depressed", ax=ax))

ax1 = df1.plot.scatter(x="subj_scores", y="compound", color="DarkRed", label="Class: Happy")

ax2 = df2.plot.scatter(x="subj_scores", y="compound", color="DarkGreen", label="Class: Depressed")

print(ax1)
print(ax2)
