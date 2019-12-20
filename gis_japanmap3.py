
!wget https://www.post.japanpost.jp/zipcode/dl/oogaki/zip/ken_all.zip

!unzip ken_all.zip

import numpy as np
import pandas as pd

filename = "KEN_ALL.CSV"
ken = pd.read_csv(filename, encoding="cp932", header=None, dtype="object")
ken.columns = ["v{0}".format(i) for i in range(15)]

ken.shape

ken.head()

!pip install geopandas

import geopandas as gpd

filename = "/content/drive/My Drive/Colab Notebooks/N03-19_190101.geojson"
gdf = gpd.read_file(filename, encoding="utf-8")

gdf.shape

gdf.head()

gdf_city = gdf.dissolve(by="N03_007", as_index=False)

gdf_city.shape

gdf_city.head()

tmp = [i is None for i in gdf_city["N03_007"]]
sum(tmp)



gdf_pref = gdf_city.dissolve(by="N03_001", as_index=False)

gdf_pref.shape

gdf_pref.head()

import pickle

with open('/content/drive/My Drive/Colab Notebooks/gdf_city.pickle', mode='wb') as f:
    pickle.dump(gdf_city, f)

with open('/content/drive/My Drive/Colab Notebooks/gdf_pref.pickle', mode='wb') as f:
    pickle.dump(gdf_pref, f)

"""## 途中からスタート"""

with open('/content/drive/My Drive/Colab Notebooks/gdf_city.pickle', mode='rb') as f:
    gdf_city = pickle.load(f)

with open('/content/drive/My Drive/Colab Notebooks/gdf_pref.pickle', mode='rb') as f:
    gdf_pref = pickle.load(f)





gdf_pref_simple = gdf_pref.copy()
gdf_pref_simple["geometry"] = gdf_pref.geometry.simplify(tolerance=0.1, preserve_topology=True)

#toleranceが１に近いほどシンプルに

gdf_pref_simple.head()

gdf_pref_simple["num"] = [i/47 for i in range(47)]



import matplotlib.pyplot as plt

gdf_pref_simple.plot()
plt.show()



gdf_pref.plot()
plt.show()

"""## Plotly"""

tmp_gdf = gdf_city.copy()
tmp_gdf = tmp_gdf[tmp_gdf["N03_001"]=="北海道"]
idx = [ti.type=="Polygon" for ti in tmp_gdf["geometry"]]
tmp_gdf = tmp_gdf.iloc[idx,:]

n=tmp_gdf.shape[0]
tmp_gdf["num"] = [i/n for i in range(n)]
n

tmp_gdf.head()

df.head()

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})

import plotly.graph_objects as go

fig = go.Figure(go.Choroplethmapbox(geojson=counties, locations=df.fips, z=df.unemp,
                                    colorscale="Viridis", zmin=0, zmax=12,
                                    marker_opacity=0.5, marker_line_width=0))
fig.update_layout(mapbox_style="carto-positron",
                  mapbox_zoom=3, mapbox_center = {"lat": 37.0902, "lon": -95.7129})
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

geo_source["features"][1]

tmp_gdf["id"] = list(range(tmp_gdf.shape[0]))

from urllib.request import urlopen
import json
import plotly.graph_objects as go

tmp_gdf["geometry"].to_file("tmp_gdf.geojson", driver='GeoJSON')

with open('tmp_gdf.geojson', 'r') as f:
    geo_source = json.load(f)

geo_source = tmp_gdf.to_json()

fig = go.Figure(go.Choroplethmapbox(geojson=geo_source, locations=tmp_gdf["id"], z=tmp_gdf["num"],
                                    colorscale="Viridis", zmin=0, zmax=1,
                                    marker_opacity=0.5, marker_line_width=0))
fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=5, mapbox_center = {"lat": 43.06417, "lon": 141.34694}) #43.06417,141.34694
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()









"""## Bokeh"""

from bokeh.plotting import figure, output_file, show, reset_output
from bokeh.models import GeoJSONDataSource, HoverTool
from bokeh.palettes import viridis
from bokeh.models import ColorBar, LogColorMapper



tmp_gdf = gdf_city.copy()
tmp_gdf = tmp_gdf[tmp_gdf["N03_001"]=="北海道"]
#idx = [ti.type=="Polygon" for ti in tmp_gdf["geometry"]]
#tmp_gdf = tmp_gdf.iloc[idx,:]

n=tmp_gdf.shape[0]
tmp_gdf["num"] = [i/n for i in range(n)]
n

tmp_gdf.to_file("tmp_gdf.geojson", driver='GeoJSON')

#https://stackoverflow.com/questions/40226189/bokeh-is-not-rendering-properly-multipolygon-islands-from-geojson
with open(r'tmp_gdf.geojson', 'r') as f:
    geo_source = GeoJSONDataSource(geojson=f.read())


#geo_source = GeoJSONDataSource(geojson=tmp_gdf.to_json())
color_mapper = LogColorMapper(viridis(256))

# 出力設定
reset_output()
output_file("graph8.html")

fig = figure(title='Number of rides')
fig.patches(xs='xs', ys='ys', alpha=0.9, source=geo_source, 
            color={'field': 'num', 'transform': color_mapper},
            line_width=1, line_alpha=0.5, line_color='black')

hover = HoverTool(
    point_policy='follow_mouse'
)
fig.add_tools(hover)

color_bar = ColorBar(color_mapper=color_mapper, 
                     location=(0, 0), 
                     label_standoff=12)
fig.add_layout(color_bar, 'right')

# グラフ表示
show(fig)



reset_output()

output_file("graph2.html")

TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]

# グラフ設定
fig = figure(title='Number of rides')

# プロット
fig.line(x, y1, legend="sin")
fig.line(x, y2, legend="cos")

# 凡例をクリックしたときにプロットを非表示にする
fig.legend.click_policy = "hide"

# グラフ表示
show(fig)

from bokeh.io import show
from bokeh.models import LogColorMapper
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure

from bokeh.sampledata.us_counties import data as counties
from bokeh.sampledata.unemployment import data as unemployment

palette.reverse()

counties = {
    code: county for code, county in counties.items() if county["state"] == "tx"
}

county_xs = [county["lons"] for county in counties.values()]
county_ys = [county["lats"] for county in counties.values()]

county_names = [county['name'] for county in counties.values()]
county_rates = [unemployment[county_id] for county_id in counties]
color_mapper = LogColorMapper(palette=palette)

data=dict(
    x=county_xs,
    y=county_ys,
    name=county_names,
    rate=county_rates,
)

TOOLS = "pan,wheel_zoom,reset,hover,save"

p = figure(
    title="Texas Unemployment, 2009", tools=TOOLS,
    tooltips=[
        ("Name", "@N03_001"), ("Unemployment rate)", "@num%"))
    ])
p.grid.grid_line_color = None
p.hover.point_policy = "follow_mouse"

p.patches('x', 'y', source=geo_source,
          fill_color={'field': 'num', 'transform': color_mapper},
          fill_alpha=0.7, line_color="white", line_width=0.5)

show(p)

















!pip install japanmap

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.colors
import matplotlib.pyplot as plt





from japanmap import *
pref_names[3]

pref_code('京都府')

plt.imshow(picture())

plt.figure(figsize=(10,20))
plt.imshow(picture({'北海道': 'blue'}))
plt.show()

df = pd.read_csv("how_many_cars.csv")

df=df.iloc[:53,:8]

df.head()

num_dict={}

for k,n in zip(df["運輸支局"], df["乗用車"]):
    if k in ["札幌", "函館", "旭川", "室蘭", "釧路", "帯広", "北見"]:
        tmp=1
    else:
        tmp = pref_code(k)
    tmp = pref_names[tmp]
    #print(k,tmp)
    if tmp not in num_dict:
        num_dict[tmp] = n
    else:
        num_dict[tmp] += n

num_dict

n_min = min(num_dict.values())
n_max = max(num_dict.values())

print(n_min)
print(n_max)

cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(vmin=n_min, vmax=n_max)

def color_scale(r):
    #return (255-(r-n_min)/(n_max-n_min)*255, 150, 255)
    tmp = cmap(norm(r))
    return (tmp[0]*255, tmp[1]*255, tmp[2]*255)

for k,v in num_dict.items():
    num_dict[k] = color_scale(v)

plt.figure(figsize=(10,8))
plt.imshow(picture(num_dict))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm)
plt.title("How many cars ?")
plt.show()

