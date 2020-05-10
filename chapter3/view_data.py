import pandas
import webbrowser
import os

data_table= pandas.read_csv("ml_house_data_set.csv")

html = data_table[0:100].to_html()

with open("data.html", "w") as f:
  f.write(html)

full_filename=os.path.abspath("data.html")
webbrowser.open("file://{}".format(full_filename))

