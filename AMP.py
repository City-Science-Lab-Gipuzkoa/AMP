"""
Google Colab:
#%pip install osmnx
#%pip install dash
#%pip install dash_leaflet
#%pip install dash_bootstrap_components
#%pip install GDAL # install GDAL first and then rasterio (3.4.0?)
#%pip install rasterio
#%pip install peartree
#%pip install diskcache
###%pip install "dash[diskcache]"
#%pip install multiprocess
#%pip install psutil
#%pip install pandana
#%pip install bs4
"""

"""
With Anaconda in a local pc, in your environment, run: ########################
note: 

1) clone the base environment:

conda create --name NameNewEnv --clone base

It comes with many libraries already installed:
example:
Gdal 
rasterio

2) conda activate NameNewEnv

3) 
conda config --add channels conda-forge
conda config --set channel_priority strict

4) Install missing libraries:
conda install pandas==1.5.1
conda install osmnx (1.9.3?)
conda install pandana 
conda install dash
conda install diskcache 
conda install bs4 
conda install multiprocess 
conda install geopy 
conda install shapely

pip install dash-leaflet
pip install dash-bootstrap-components
pip install peartree
#pip install dash-loading-spinners
#pip install lxml (already installed in base env)
###############################################################################
"""


import sys  # Import the sys module to access system-specific parameters and functions
# Print the Python version to the console
print("Python version")
# Use the sys.version attribute to get the Python version and print it
print(sys.version)
# Print information about the Python version
print("Version info.")
# Use the sys.version_info attribute to get detailed version information and print it
print(sys.version_info)

import dash
import dash_bootstrap_components as dbc
#import dash_html_components as html
from dash import html
from dash import dcc, Output, Input, State, callback
import dash_leaflet as dl
from dash.long_callback import DiskcacheLongCallbackManager

from shapely.geometry import mapping
from shapely import MultiPoint, concave_hull
#from shapely.geometry import MultiPoint

from shapely.geometry import Point

import osmnx as ox
import networkx as nx
#import json
import geopandas as gpd
import numpy as np
#from google.colab import drive
import rasterio
import math
import os
import time
import json
import peartree as pt
import datetime
import geopy.distance
import zipfile

import plotly.express as px
import pandas as pd
import pandana
from pandana.loaders import osm

import requests

## Diskcache
import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

#drive.mount('/content/drive',  force_remount=True)

root_dir = 'C:/Users/gfotidellaf/repositories/AMP/assets/'
sys.path.append(root_dir + 'modules')

"""
print('osmnx version')
print(ox.__version__)
ox.config(log_console=True, log_file=True, use_cache=False)
settings = '[out:json][timeout:1800]'
#ox.settings.overpass_settings = settings.format(year = year)
ox.settings.overpass_settings = settings
ox.settings.overpass_rate_limit = False
ox.settings.requests_timeout = 100000
ox.settings.overpass_memory = 1000000000
"""

center = [43.268593943060836, -1.9741267301558392]
#image_path = 'assets/CSL.PNG'
#im1 = '/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/CSL_logo.PNG'
#im2 = '/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/DFG_logo.png'
im1 = root_dir +'images/CSL_logo.PNG'
im2 = root_dir +'images/DFG_logo.png'
im3 = root_dir +'images/MUBIL_logo.png'

from PIL import Image
image1 = Image.open(im1)
image2 = Image.open(im2)
image3 = Image.open(im3)


# raster data from: https://srtm.csi.cgiar.org/srtmdata/
#raster_path = '/home/beppe23/mysite/assets/srtm_36_04.tif'
#raster_path = '/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/Accessibility_Map/srtm_36_04.tif'
#raster_path = '/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/Accessibility_Map/srtm_36_04.zip'
raster_path = root_dir + 'data/srtm_36_04.tif'


#Require_file = "/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/Accessibility_Map/Python_Requirements.txt"
#Require_file = "/home/beppe23/mysite/assets/Python_Requirements.txt"
Require_file = root_dir + "Python_Requirements.txt"


# GTFS files from: https://transitfeeds.com/p/euskotren/655
#GTFS_path = '/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/gtfs_Euskotren.zip'
#GTFS_path = '/content/drive/MyDrive/Colab Notebooks/CSL_GIPUZKOA/20240329_130132_Euskadi_Euskotren.zip'
#GTFS_path = '/home/beppe23/mysite/assets/gtfs_Euskotren.zip'
#GTFS_path = '/home/beppe23/mysite/assets/20240329_130132_Euskadi_Euskotren.zip'
GTFS_path = root_dir + 'data/20240329_130132_Euskadi_Euskotren.zip'


# Walk speed: Using crowdsourced fitness tracker data to model the relationship between slope and travel rates
# https://doi.org/10.1016/j.apgeog.2019.03.008


#!pip freeze > Require_file
#!pip freeze > '{Require_file}' # Colab
#os.system('pip freeze > ' +Require_file) # server
os.system('pip list > ' +Require_file) # server


# get OpenStreetMap Tags #######################################################
from bs4 import BeautifulSoup # library to parse HTML documents
# get the response in the form of html
wikiurl="https://wiki.openstreetmap.org/wiki/OpenStreetMap_Carto/Symbols"
table_class="wikitable"
response=requests.get(wikiurl)
print(response.status_code)
# parse data from the html into a beautifulsoup object
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find('table',{'class':"wikitable"})
df=pd.read_html(str(table))
# convert list to dataframe
df=pd.DataFrame(df[0])
#print(df.head(20))
#print(df[['Tags']])
Tags =[]
for i in range(len(df)):
    tmp = df.loc[i,'Tags'][0]
    if "=" in tmp:
       if "/" in tmp:
          tmp1 = tmp.split('/')
          for j in tmp1:
             #print('composed:')
             #print(j)
             Tags.append(j)
       else:
          #print(tmp)
          Tags.append(tmp)
       #print()
#print(Tags[:4])
Tags_dropdown = dcc.Dropdown(Tags, multi=False, style={"margin-top": "15px"},id='tags_dropdown')
################################################################################


"""
def bbox_coords(lat0,lon0,d):
    r_earth = 6378 # km
    dx = d*np.cos(45*np.pi/180)
    dy = d*np.sin(45*np.pi/180)
    lat_max = lat0 + (dy / r_earth) * (180 / np.pi)
    lon_max = lon0 - (dx / r_earth) * (180 / np.pi) / np.cos(lat0 * np.pi/180)
    lat_min = lat0 - (dy / r_earth) * (180 / np.pi)
    lon_min = lon0 + (dx / r_earth) * (180 / np.pi) / np.cos(lat0 * np.pi/180)
    return [lat_max, lat_min, lon_max, lon_min]
"""

def connect_nn(gr,cl):
   cutoff = 0.035 #km
   x1 = gr.nodes[cl]['x']
   y1 = gr.nodes[cl]['y']
   #cl = ox.nearest_nodes(gr, L[0],L[1])
   coords_1 = (y1,x1)
   # Find nodes within cutoff distance: #######################################################
   #for i, node1 in gr.nodes(data=True):
   gdf_nodes = ox.graph_to_gdfs(gr, edges=False)[['geometry']]
   # create array of points
   coords = np.array([coords_1])
   # get buffers around each point at a distance = cutoff
   points = gpd.GeoDataFrame(crs='epsg:4326', geometry=gpd.points_from_xy(coords[:, 1], coords[:, 0]))
   buffers = ox.project_gdf(ox.project_gdf(points).buffer(cutoff*1000), to_latlong=True)
   gdf_buffers = gpd.GeoDataFrame(geometry=buffers)
   # find all the nodes within the buffer of each point
   result = gpd.sjoin(gdf_buffers, gdf_nodes, how='left', predicate='intersects')['index_right']
   ############################################################################################
   # Loop over nodes within cutoff distance: ##################################################
   for i in result:
       node1 = gr.nodes[i]
       coords_2 = (node1['y'],node1['x'])
       d = geopy.distance.geodesic(coords_1, coords_2).km
       if (d < cutoff) and (d > 0.0):
             weight = "time"
             #weight = "length"
             # find shortest path
             route = nx.shortest_path(gr, i, cl, weight)
             #gdf = ox.utils_graph.route_to_gdf(gr, route, weight)
             gdf = ox.routing.route_to_gdf(gr, route, weight)
             #duration = gdf["length"].sum()
             duration = gdf["time"].sum()
             if duration > 10.0:
                   gr.add_edges_from([(i,cl),(cl,i)], time=0.3, length=10.0, boarding_cost=0.0, highway= 'footway', oneway= False, reversed =False)
                   print(i,cl,coords_1, coords_2,'distance: ',d)
                   print('edge added! Time larger:',duration)
   return gr


def reduce_distance(gr,ori_node,dest_node,ws,tt):
   weight = "time"
   #weight = "length"
   # find shortest path
   route = nx.shortest_path(gr, ori_node, dest_node, weight)
   #gdf = ox.utils_graph.route_to_gdf(gr, route, weight)
   gdf = ox.routing.route_to_gdf(gr, route, weight)
   #duration = gdf["length"].sum()
   duration = gdf["time"].sum()
   if duration < tt:
      return (ws*1000/60)*(tt - duration)
   else:
      return (ws*1000/60)*1


#app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
#app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)




# Isochronic Map DropDown Menu Options##########################################
level3_items = [
    dbc.DropdownMenuItem("Euskotren", id="Eusko")
]
level2_items = [
    dbc.DropdownMenuItem("constant speed", id="constant"),
    dbc.DropdownMenuItem("Tobler", id="Tob"),
    dbc.DropdownMenuItem("Irmischer-Clarke", id="I-C"),
    dbc.DropdownMenuItem("Rees", id="Re")
]
level1_items = [
    html.Div(
        id="submenu",
        children=[
            dbc.DropdownMenuItem("Drive", id="Dri"),
            dbc.DropdownMenuItem("Bike", id="Bik"),
            dbc.DropdownMenu(level2_items,direction="end", label='Walk'),
            dbc.DropdownMenu(level3_items,direction="end", label='Public Transit'),
        ],
    )
]
################################################################################



# Accessibility Map DropDown Menu Options#######################################
level1_items_Acc_ind = [
            dbc.DropdownMenuItem("Reach", id="Rea"),
            dbc.DropdownMenuItem("Gravity", id="Gra"),
            dbc.DropdownMenuItem("KNN", id="Knn"),
            dbc.DropdownMenuItem("Custom", id="Cus")
]
level1_items_DistanceTo = [
            dbc.DropdownMenuItem("hospital", id="Hosp"),
            dbc.DropdownMenuItem("bus_stop", id="PT"),
            dbc.DropdownMenuItem("school", id = "Scho"),
            dbc.DropdownMenuItem("bar", id = "Bar"),
            dbc.DropdownMenuItem("restaurant", id = "Rest"),
            dbc.DropdownMenuItem("marketplace", id = "Mark")
]

level1_items_acc = [
    html.Div(
        id="submenu_acc_var",
        children=[
            dbc.DropdownMenu(level1_items_DistanceTo,direction="end", label='Distance to'),
            dbc.DropdownMenu(level1_items_Acc_ind,direction="end", label='Combined')
        ],
    )
]

"""
level1_items_acc_ind = [
    html.Div(
        id="submenu_acc_ind",
        children=[
            dbc.DropdownMenuItem("Reach", id="Rea"),
            dbc.DropdownMenuItem("Gravity", id="Gra"),
            dbc.DropdownMenuItem("KNN", id="Knn"),
            dbc.DropdownMenuItem("Custom", id="Cus")
        ],
    )
]
"""
################################################################################

sidebarIso = html.Div(
    [
      html.P([ html.Br(),'Choose transport mode'],style={"font-weight": "bold"}),
      dbc.DropdownMenu(label="Menu", children=level1_items,id="dropdown_TransOpt"),
      html.P([ html.Br(),'Time for Isochronic Map (in min)'],style={"font-weight": "bold"}),
      dcc.Input(id="choose_time", type="text", placeholder="", style={'marginRight':'10px','width': 50,'height': 20}),
      html.P([ html.Br(),'Walk speed at 0 slope (km/hour)']),
      dcc.Input(id="choose_walk_speed", type="text", value='4.5', placeholder="", style={'marginRight':'10px','width': 50,'height': 20}),
      html.P([ html.Br(),'Concave hull ratio (0-1)'],id='concave_hull_Iso'),
      dcc.Input(id="choose_ch_ratio_Iso", type="text", value='0.255', placeholder="", style={'marginRight':'10px','width': 50,'height': 20}),
      dbc.Popover('Optimum value might change\n based on tranport mode!',
                  target="choose_ch_ratio_Iso",
                  body=True,
                  trigger="hover"),
      html.Div(id='out_text_Iso')
    ],id='sidebar_id_Iso'
)

test_polygon = dl.Polygon(positions=[[43.31115774122127, -2.013441259792672], [43.29540055344163, -1.9974906467481823], [43.30086350770776, -1.9851487696866084], [43.31509640001811, -1.999944587216565]])
keys = ["Isochronic map", "Accessibility map"]

contentIso = html.Div([
    html.Div([
        html.Img(src=image1,style={'width':'40%', "display": "inlineBlock", "verticalAlign": "top"}),
        html.Img(src=image2,style={'width':'25%',"display": "inlineBlock", "verticalAlign": "top"}),
        html.Img(src=image3,style={'width':'25%',"display": "inlineBlock", "verticalAlign": "top"}),
        dbc.Spinner(html.Div(id="loading-output_Iso"), color="primary", spinner_style={'zIndex': 999, 'position': 'absolute','left':'300px','top':'150px','width': '10rem', 'height': '10rem'})
         ],
         style= {'verticalAlign': 'top'}),
    html.Div([
         dl.Map([
             dl.LayersControl(
               [dl.BaseLayer(dl.TileLayer(), name=key, checked=key== "Isochronic map") for key in keys]+
               [dl.Polygon(positions=[], id='position-data_Iso'),
                dl.Polygon(positions=[], id='acc-position-data_Iso'),
                dl.ScaleControl(position="bottomright"),
                dl.Overlay(dl.LayerGroup(test_polygon), name="test_polygon_Iso", checked=False)], id="lc_Iso")],
         id='mapa_Iso',
         center=center, zoom=14, style={'height': '80vh','margin-top':10, 'cursor': 'pointer'})],
         id='outer_Iso')
    ])


tab1 = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebarIso, width=3, className='bg-light'),
                dbc.Col(contentIso, width=9)
                ],
            style={"height": "100vh"}
            ),
    ],
    fluid=True
)


#       html.P([ html.Br(),'Choose Accessibility Metrics'],style={"font-weight": "bold"}),
#       dbc.DropdownMenu(label="Menu", children=level1_items_acc,id="dropdown_AccOpt"),
"""
       dcc.Input(id="choose_POI_tag", type="text", value='shop=supermarket', placeholder="", style={'marginRight':'10px','width': 150,'height': 50}),
       dbc.Popover('See https://wiki.openstreetmap.org/wiki/OpenStreetMap_Carto/Symbols',
                  target="choose_POI_tag",
                  body=True,
                  trigger="hover"),
"""
sidebarAcc = html.Div(
       [
       html.P([ html.Br(),'Select POI\'s tag'],id='POI_tag',style={"font-weight": "bold"}),
       Tags_dropdown,
       html.Div(id='out_text_Acc')
       ],id='sidebar_id_Acc')


"""
contentAcc = html.Div([
    html.Div([
         html.Img(src=image_path, style={'height':'30%', 'width':'30%'}),
         dbc.Spinner(html.Div(id="loading-output_Acc"), color="primary", spinner_style={'zIndex': 999, 'position': 'absolute','left':'300px','top':'150px','width': '10rem', 'height': '10rem'})
         ],
         style= {'verticalAlign': 'top'}),
    html.Div([
         dl.Map([
             dl.LayersControl(
               [dl.BaseLayer(dl.TileLayer(), name=key, checked=key== "Accessibility map") for key in keys]+
               [dl.Polygon(positions=[], id='position-data_Acc'),
                dl.Polygon(positions=[], id='acc-position-data_Acc'),
                dl.ScaleControl(position="bottomright"),
                dl.Overlay(dl.LayerGroup(test_polygon), name="test_polygon_Acc", checked=False)], id="lc_Acc")],
         id='mapa_Acc',
         center=center, zoom=14, style={'height': '80vh','margin-top':10, 'cursor': 'pointer'})],
         id='outer_Acc')
    ])
"""

# Initialize Figure (not the map!) #############################################
#bbox = [43.25979205399755, -2.037454224442623, 43.277217052603206, -1.9945668200525066]
bbox = [43.26358154899398, -1.9849938514898302, 43.27242459264855, -1.9593014827641313]
G = ox.graph_from_bbox(bbox[0], bbox[2], bbox[1], bbox[3],network_type='walk', simplify=False)
nodes_df = ox.graph_to_gdfs(G, nodes=True, edges=False)
edges_df = ox.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=True)
edges_df = edges_df.reset_index()
network = pandana.Network(nodes_df['x'], nodes_df['y'],
                   edges_df['u'], edges_df['v'],
                   edges_df[['length']])
d = {'Lat': network.nodes_df.y, 'Lon': network.nodes_df.x}
df = pd.DataFrame(data=d,columns=['Lat', 'Lon'])
fig = px.scatter_mapbox(df,
                           center={"lat": center[0], "lon": center[1]},
                           mapbox_style = 'open-street-map', zoom=14)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
################################################################################
#         html.Img(src=image_path, style={'height':'30%', 'width':'30%'}),

contentAcc = html.Div([
    html.Div([
         html.Img(src=image1,style={'width':'50%', "display": "inlineBlock", "verticalAlign": "top"}),
         html.Img(src=image2,style={'width':'30%',"display": "inlineBlock", "verticalAlign": "top"}),
         dbc.Spinner(html.Div(id="loading-output_Acc"), color="primary", spinner_style={'zIndex': 999, 'position': 'absolute','left':'300px','top':'150px','width': '10rem', 'height': '10rem'})
         ],
         style= {'verticalAlign': 'top'}),
    html.Div([
           dl.Map(
                  [dl.TileLayer(), dl.Polygon(positions=[], id='position-data_Acc')],
                 id='mapa_Acc',
                 center=center, zoom=14, style={'height': '50vh','margin-top':10, 'cursor': 'pointer'}
          )
      ]),
    dcc.Graph(id = 'AccFig', figure=fig),
    html.Div(id='out_text_Acc')
])

tab2 = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebarAcc, width=3, className='bg-light'),
                dbc.Col(contentAcc, width=9)
                ],
            style={"height": "100vh"}
            ),
    ],
    fluid=True
)


app.layout = html.Div([
    html.H1('CSL@Gipuzkoa: Mapping Accessibility'),
    dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        dcc.Tab(id="tab-1", label='Isochronic Map', value='tab-1-example'),
        dcc.Tab(id="tab-2", label='Accessibility Map', value='tab-2-example'),
    ]),
    html.Div(id='tabs-content-example',
             children = tab1)
])

@app.callback(dash.dependencies.Output('tabs-content-example', 'children'),
             [dash.dependencies.Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1-example':
        return tab1
    elif tab == 'tab-2-example':
        return tab2

@app.callback(
           dash.dependencies.Output("dropdown_TransOpt", "label"),
           dash.dependencies.Input("Dri", "n_clicks"),
           dash.dependencies.Input("Bik", "n_clicks"),
           dash.dependencies.Input("constant", "n_clicks"),
           dash.dependencies.Input("Tob", "n_clicks"),
           dash.dependencies.Input("I-C", "n_clicks"),
           dash.dependencies.Input("Re", "n_clicks"),
           dash.dependencies.Input("Eusko", "n_clicks")
       )
def update_labelIso(n1, n2, n3, n4 , n5, n6, n7):
           # use a dictionary to map ids back to the desired label
           # makes more sense when there are lots of possible labels
           id_lookup = {
               "Dri": "drive",
               "Bik": "bike",
               "constant": "constant speed",
               "Tob"  : "Tobler",
               "I-C": "Irmischer-Clarke",
               "Re": "Rees",
               "Eusko": "Euskotren", }

           ctx = dash.callback_context

           if (n1 is None and n2 is None and n3 is None and n4 is None and n5 is None and n6 is None and n7 is None) or not ctx.triggered:
               # if neither button has been clicked, return "Not selected"
               return "Not selected"

           # this gets the id of the button that triggered the callback
           button_id = ctx.triggered[0]["prop_id"].split(".")[0]
           return id_lookup[button_id]
           #return 'selected?'


@app.callback([dash.dependencies.Output('position-data_Iso', 'positions'),dash.dependencies.Output("loading-output_Iso", "children")],
                     dash.dependencies.Input('mapa_Iso', 'clickData'),
                     dash.dependencies.Input("lc_Iso", "baseLayer"),
                     dash.dependencies.State('dropdown_TransOpt', 'label'),  # State does not trigger the callback
                     dash.dependencies.State('choose_time', 'value'),
                     dash.dependencies.State('choose_walk_speed', 'value'),
                     dash.dependencies.State('choose_ch_ratio_Iso', 'value'),
                     prevent_initial_call=True)
def on_click(coord,base_layer,opt,t,wlk_sp,ch_ratio):
           #try:
           if (coord is not None):
               Lat = coord['latlng']['lat']
               Lon = coord['latlng']['lng']
               map_center = (Lat, Lon)
               #distance = float(max_dist)
               trip_time = float(t)
               walk_speed = float(wlk_sp)
               distance = (walk_speed * 1000 / 60) * trip_time # in meters
               chratio = float(ch_ratio)
               walk_opts = ['constant speed', 'Tobler', 'Irmischer-Clarke', 'Rees', 'Euskotren']

               if opt in walk_opts:
                   G = ox.graph_from_point(map_center, dist=distance, network_type='walk', simplify=False)
               else:
                   G = ox.graph_from_point(map_center, dist=distance, network_type=opt, simplify=False)

               if opt == 'bike':
                  travel_speed = 20 # km/hour
                  meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
                  for u, v, k, data in G.edges(data=True, keys=True):
                      data['time'] = data['length'] / meters_per_minute

               if opt == 'drive':
                  # fill in edges with missing `maxspeed` from OSM (km/hour)
                  hwy_speeds = {"residential": 35, "secondary": 50, "tertiary": 60}
                  G = ox.add_edge_speeds(G, hwy_speeds)
                  G = ox.add_edge_travel_times(G)
                  for u, v, k, data in G.edges(data=True, keys=True):
                      #data['time'] = data['length'] / (data['maxspeed'] *1000 / 60)
                      data['time'] = data['length'] / (data['speed_kph'] *1000 / 60)

               if opt == 'constant speed':
                  travel_speed = walk_speed # km/hour
                  meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
                  for u, v, k, data in G.edges(data=True, keys=True):
                      data['time'] = data['length'] / meters_per_minute

               if opt == 'Tobler':
                  #raster_path = '/home/beppe23/mysite/assets/srtm_36_04.tif'
                  G = ox.elevation.add_node_elevations_raster(G, raster_path, cpus=1)
                  G = ox.elevation.add_edge_grades(G, add_absolute=False)
                  for u, v, k, data in G.edges(data=True, keys=True):
                      if(math.isnan(data['grade']) or math.isinf(data['grade'])):
                           travel_speed = walk_speed
                      else:
                           d0 = walk_speed/np.exp(-3.5 * abs(0.0 + 0.05))
                           travel_speed = d0*np.exp(-3.5 * abs(data['grade'] + 0.05)) # km/hour
                      meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
                      data['time'] = data['length'] / meters_per_minute
               if opt == 'Irmischer-Clarke':
                  #raster_path = '/home/beppe23/mysite/assets/srtm_36_04.tif'
                  G = ox.elevation.add_node_elevations_raster(G, raster_path, cpus=1)
                  G = ox.elevation.add_edge_grades(G, add_absolute=False)
                  for u, v, k, data in G.edges(data=True, keys=True):
                      if(math.isnan(data['grade']) or math.isinf(data['grade'])):
                           travel_speed = walk_speed
                      else:
                           d0 = walk_speed - (0.11 + np.exp(-(1/1800)*(100 * np.tan(0.0) + 5)**2) ) * 10**-3*3600
                           travel_speed = d0 + (0.11 + np.exp(-(1/1800)*(100 * np.tan(data['grade']) + 5)**2) ) * 10**-3*3600 # km/hour
                           meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
                           data['time'] = data['length'] / meters_per_minute

               if opt == 'Rees':
                  #raster_path = '/home/beppe23/mysite/assets/srtm_36_04.tif'
                  G = ox.elevation.add_node_elevations_raster(G, raster_path, cpus=1)
                  G = ox.elevation.add_edge_grades(G, add_absolute=False)
                  for u, v, k, data in G.edges(data=True, keys=True):
                       if(math.isnan(data['grade']) or math.isinf(data['grade'])):
                           travel_speed = walk_speed
                       else:
                           d0 = walk_speed - ( 1/(0.75 + 0.09 * np.tan(0.0+0.05) + 14.6 * np.tan(0.0+0.05)**2 ) ) * 10**-3 * 3600
                           travel_speed = d0 + (1/(0.75 + 0.09*np.tan(data['grade']+0.05)+ 14.6*np.tan(data['grade']+0.05)**2))*10**-3*3600 # km/hour
                           if travel_speed > 0:
                              meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
                           else:
                              meters_per_minute = 0.0 * 1000 / 60 #km per hour to m per minute
                       data['time'] = data['length'] / meters_per_minute

               if opt == 'Euskotren':
                  print('inside Euskotren')
                  now = datetime.datetime.now()
                  start = now.hour * 60 * 60 + now.minute * 60
                  end = start + trip_time * 60 # 8 * 60 * 60
                  #end = start + 10 * 60 * 60 # 8 * 60 * 60
                  #start = 7 * 60 * 60
                  #end =   9 * 60 * 60
                  travel_speed = walk_speed # km/hour
                  meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
                  feed = pt.get_representative_feed(GTFS_path)
                  G0 = pt.load_feed_as_graph(feed, start, end)
                  print('transport graph generated!')
                  #G0 = G0.to_directed() # does this symmetrize the graph edges?
                  #for u, v, k, data in G.edges(data=True, keys=True):
                  #    data['time'] = data['length'] / 60
                  edges_PT = []
                  for from_node, to_node, edge in G0.edges(data=True):
                      orig_len = edge['length']
                      G0[from_node][to_node][0]['length'] = orig_len/60
                      G0[from_node][to_node][0]['time'] = orig_len/60
                      edges_PT.append([from_node,to_node,edge])

                  print('Edge length and time calculated!')
                  # Find and add missing edges in PT graph ######################################################
                  # GTFS files seems missing some edge. This piece of code should fix this problem ##############
                  #edges_PT = []
                  #for u, v, k, data in G0.edges(data=True, keys=True):
                  #    edges_PT.append([u,v,data])
                  for u, v, k, data in G0.edges(data=True, keys=True):
                      if [v,u,data] not in edges_PT:
                         G0.add_edges_from([(v,u)], length= data['length'], mode= data['mode'], time=data['time'])
                  ################################################################################################

                  # check if station is within walking distance from clicked point ##########################
                  orig_node = ox.nearest_nodes(G0, map_center[1], map_center[0])
                  d0 = [G0.nodes[orig_node]['y'],G0.nodes[orig_node]['x']]
                  d_check = geopy.distance.geodesic(d0, map_center).km
                  if d_check < 0.9*walk_speed*trip_time/60:
                     # get train network at a distance equal to "trip_time"
                     G0 = nx.ego_graph(G0, orig_node, radius=trip_time, distance='time')
                     temp_graphs = [G0]
                     nodes_to_link = []
                     # for each node (station) of the train network build a "walk" network centered at the train station
                     for i, node0 in G0.nodes(data=True):
                         #d0 = [node0['x'],node0['y']]
                         x = G0.nodes[i]['x']
                         y = G0.nodes[i]['y']
                         # make walk graph distance progressively smaller as we move away from origin station ########################
                         if i != orig_node:
                            distance_k = reduce_distance(G0,orig_node,i,walk_speed,trip_time)
                            print(distance_k)
                         else:
                            distance_k = distance
                         #############################################################################################################
                         #Gwalk_temp = ox.graph_from_point([y,x], dist=distance, network_type='walk', simplify=False)
                         Gwalk_temp = ox.graph_from_point([y,x], dist=distance_k, dist_type='bbox', network_type='walk', simplify=False)
                         #north, south, east, west = y + 0.008, y - 0.008, x + 0.008, x - 0.008
                         #north, south, east, west = bbox_coords(y,x,walk_speed*trip_time/60)
                         #print(north, south, east, west)
                         #Gwalk_temp = ox.graph_from_bbox(north, south, east, west, network_type='walk', simplify=False)
                         for u, v, k, data in Gwalk_temp.edges(data=True, keys=True):
                             data['time'] = data['length'] / meters_per_minute
                             Gwalk_temp.nodes[u]['boarding_cost'] = 0
                             Gwalk_temp.nodes[v]['boarding_cost'] = 0
                         ori = ox.nearest_nodes(Gwalk_temp, x,y)
                         Gwalk_temp = connect_nn(Gwalk_temp,ori)
                         temp_subG = nx.ego_graph(Gwalk_temp, ori, radius=trip_time, distance='time')
                         # create node pairs describing the bidirectional
                         # connection between train station "i" and walk node "ori"
                         nodes_to_link.append((i,ori)) # from the train station to walk node!
                         nodes_to_link.append((ori,i)) # from the walk node to train station!
                         temp_graphs.append(temp_subG)

                     G = nx.compose_all(temp_graphs)
                     # create the bidirectional edge connection between walk nodes and train stations
                     G.add_edges_from(nodes_to_link, time=0.3, length=10.0, boarding_cost=0.0, highway= 'footway', oneway= False, reversed =False)
                     #orig_node = ox.nearest_nodes(G, map_center[1], map_center[0])
                     # caculate the isochronic map over the full walk-public transit
                     # graph starting from clicked point
                     subgraph = nx.ego_graph(G, orig_node, radius=trip_time, distance='time')
                  else:
                      # default backup on "constant speed" walk iscronic map
                      G = ox.graph_from_point(map_center, dist=distance, network_type='walk', simplify=False)
                      for u, v, k, data in G.edges(data=True, keys=True):
                          data['time'] = data['length'] / meters_per_minute
                      orig_node = ox.nearest_nodes(G, map_center[1], map_center[0])
                      subgraph = nx.ego_graph(G, orig_node, radius=trip_time, distance='time')

                  #G = nx.compose_all([G,Gwalk])
               # add an edge attribute for time in minutes required to traverse each edge
               # get closes graph nodes to origin and destination
               if opt!= 'Euskotren':
                  orig_node = ox.nearest_nodes(G, map_center[1], map_center[0])
                  subgraph    = nx.ego_graph(G, orig_node, radius=trip_time, distance='time')
               node_points = [Point(data["x"], data["y"]) for node, data in subgraph.nodes(data=True)]
               #node_points = [[data["y"],data["x"]] for node, data in subgraph.nodes(data=True)]
               #iso_points = gpd.GeoSeries(node_points).unary_union.convex_hull
               mpt = MultiPoint(node_points) # just for Concave hull
               iso_points = concave_hull(mpt, ratio=chratio) # just for Concave hull
               poly_mapped = mapping(iso_points)
               poly_coordinates = poly_mapped['coordinates'][0]
               poly = [ [coords[1], coords[0]] for coords in poly_coordinates]

               #polys.append(poly)
               #return polys[0]
               # clear cache and other temp files in server: ######################
               os.system("rm -rf /tmp/.*")
               os.system("rm -rf /home/beppe23/cache/*.json")
               os.system("rm -rf ~/.cache")
               ####################################################################

               return [poly,True]
               #return json.dumps(polys[0])

           else:
              #return {}
              return [[],False]



@app.callback(
           dash.dependencies.Output("dropdown_AccOpt", "label"),
           dash.dependencies.Input("Hosp", "n_clicks"),
           dash.dependencies.Input("PT", "n_clicks"),
           dash.dependencies.Input("Scho", "n_clicks"),
           dash.dependencies.Input("Bar", "n_clicks"),
           dash.dependencies.Input("Rest", "n_clicks"),
           dash.dependencies.Input("Mark", "n_clicks"),
           dash.dependencies.Input("Rea", "n_clicks"),
           dash.dependencies.Input("Gra", "n_clicks"),
           dash.dependencies.Input("Knn", "n_clicks"),
           dash.dependencies.Input("Cus", "n_clicks")
           )
def update_labelAcc_Ind(n1, n2, n3, n4 , n5, n6, n7, n8, n9, n10):
           # use a dictionary to map ids back to the desired label
           # makes more sense when there are lots of possible labels
           id_lookup = {
               "Hosp": "hospital",
               "PT": "bus_stop",
               "Scho": "school",
               "Bar"  : "bar",
               "Rest"  : "restaurant",
               "Mark"  : "marketplace",
               "Rea": "Reach",
               "Gra": "Gravity",
               "Knn": "KNN",
               "Cus"  : "Custom"
               }

           ctx = dash.callback_context

           if (n1 is None and n2 is None and n3 is None and n4 is None and n5 is None and n6 is None and n7 is None and n8 is None and n9 is None and n10 is None) or not ctx.triggered:
               # if neither button has been clicked, return "Not selected"
               return "Not selected"

           # this gets the id of the button that triggered the callback
           button_id = ctx.triggered[0]["prop_id"].split(".")[0]
           return id_lookup[button_id]


#    dash.dependencies.Output('mapa_Acc', 'children'),
#    dash.dependencies.Output('out_text_Acc', 'children'),
#     [dash.dependencies.Output('mapa_Acc', 'children'),dash.dependencies.Output("loading-output_Acc", "children")],
#@app.callback(
#    [dash.dependencies.Output('AccMap', 'figure'),dash.dependencies.Output("loading-output_Acc", "children")],
#    dash.dependencies.Input('mapa_Acc', 'clickData'),
#    dash.dependencies.State('dropdown_AccOpt', 'label'),  # State does not trigger the callback
#    prevent_initial_call=True)
"""
@app.long_callback(
    [dash.dependencies.Output('AccMap', 'figure'),dash.dependencies.Output("loading-output_Acc", "children")],
    dash.dependencies.Input('mapa_Acc', 'clickData'),
    dash.dependencies.State('dropdown_AccOpt', 'label'),  # State does not trigger the callback
    prevent_initial_call=True,
    manager=long_callback_manager)
"""
#    dash.dependencies.State('dropdown_AccOpt', 'label'),  # State does not trigger the callback
#    [dash.dependencies.Output('AccFig', 'figure'),dash.dependencies.Output("loading-output_Acc", "children")],
"""
    [dash.dependencies.Output('AccFig', 'figure'),dash.dependencies.Output("loading-output_Acc", "children")],
    dash.dependencies.State('choose_POI_tag', 'value'),  # State does not trigger the callback
    dash.dependencies.State("mapa_Acc", 'bounds'),
    dash.dependencies.Input("mapa_Acc", 'clickData'),
    prevent_initial_call=True,
    manager=long_callback_manager)

"""
#     dash.dependencies.State('choose_POI_tag', 'value'),  # State does not trigger the callback

"""
@app.long_callback(
    [dash.dependencies.Output('AccFig', 'figure'),dash.dependencies.Output("loading-output_Acc", "children")],
    dash.dependencies.State('tags_dropdown', 'value'),  # State does not trigger the callback
    dash.dependencies.State("mapa_Acc", 'bounds'),
    dash.dependencies.Input("mapa_Acc", 'clickData'),
    prevent_initial_call=True,
    manager=long_callback_manager)
"""

@app.long_callback(
    [dash.dependencies.Output('AccFig', 'figure'),dash.dependencies.Output("loading-output_Acc", "children")],
    dash.dependencies.State('tags_dropdown', 'value'),  # State does not trigger the callback
    dash.dependencies.State("mapa_Acc", 'bounds'),
    dash.dependencies.Input("mapa_Acc", 'clickData'),
    prevent_initial_call=True,
    manager=long_callback_manager)
def display(POIt,bounds,clickD):
    #amenity='restaurant'
    #amenity='marketplace'
    #amenity = 'supermarket'
    cutoff = 1000
    N_pois = 1 # nearest POI
    print('clicked point')
    Lat = clickD['latlng']['lat']
    Lon = clickD['latlng']['lng']
    print(clickD)
    print('selected bbox:')
    print(bounds)
    POIt0 = POIt.split('=')[0]
    POIt1 = POIt.split('=')[1]
    tag = '"'+POIt0+'"="' + POIt1 + '"'
    print('Selected tag:')
    print(tag)

    bbox = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]

    G = ox.graph_from_bbox(bbox[0], bbox[2], bbox[1], bbox[3],network_type='walk', simplify=False)

    #pois = osm.node_query(bbox[0], bbox[1], bbox[2], bbox[3], tags='"amenity"="{}"'.format(amenity))
    #pois = osm.node_query(bbox[0], bbox[1], bbox[2], bbox[3], tags='"shop"="{}"'.format(amenity))
    pois = osm.node_query(bbox[0], bbox[1], bbox[2], bbox[3], tags=tag)

    """
    # Smaller scale calculation ################################################
    Lat = coord['latlng']['lat']
    Lon = coord['latlng']['lng']
    center = (Lat, Lon)
    max_dist_G   = 200
    max_dist_POI = 300
    #POI = 'school'

    print('generating graph...')
    G = ox.graph_from_point(center, dist=max_dist_G, network_type='walk', simplify=False)
    tags = {'amenity': [accOpt],
           'building': True}
    lats = []
    lons = []
    distances = []
    for origin, d in G.nodes(data=True):
     origin_lat = G.nodes[origin]['y']
     origin_lon = G.nodes[origin]['x']
     oxy = (origin_lat,origin_lon)
     try:
        gdf = ox.features.features_from_point(oxy, dist=max_dist_POI, tags=tags)
        #print('POIs loaded!')
        POI_geom = gdf[gdf['amenity']==accOpt][['geometry']]
        #print('POIs geometry obtained!')
        #aver_dist = 0.0
        min_dist = 10000000
        for i in range(len(POI_geom.index)):
          if POI_geom.index[i][0]=='node':
             lat = POI_geom['geometry'][i].y
             lon = POI_geom['geometry'][i].x
          else:
             lat = POI_geom['geometry'][i].centroid.y
             lon = POI_geom['geometry'][i].centroid.x

          destination = ox.distance.nearest_nodes(G, lon, lat)
          dist = nx.shortest_path_length(G, origin, destination, weight='length')
          if dist < min_dist:
             min_dist = dist
        lat = G.nodes[origin]['y']
        lon = G.nodes[origin]['x']
        lats.append(lat)
        lons.append(lon)
        #distances.append(aver_dist)
        distances.append(min_dist)
        #nodes.append(origin)
     except:
         print('something wrong happened!')
         #pass
    print('list of coords generated!')

    d = {'Lat': lats, 'Lon': lons, 'Distance': distances}
    df = pd.DataFrame(data=d,columns=['Lat', 'Lon', 'Distance'])
    ############################################################################
    """

    #bbox = [float(bounds[0][0]), float(bounds[0][1]), float(bounds[1][0]),float(bounds[1][1])]
    ##network = osm.pdna_network_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], network_type='walk', two_way=True)
    #pois = osm.node_query(bbox[0], bbox[1], bbox[2], bbox[3], tags='"amenity"="{}"'.format(amenity))
    #G = ox.graph_from_bbox(bbox[0], bbox[2], bbox[1], bbox[3],network_type='walk', simplify=True)

    print('graph generated')
    nodes_df = ox.graph_to_gdfs(G, nodes=True, edges=False)
    print('node dataf. generated')
    edges_df = ox.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=True)
    print('edge dataf. generated')
    edges_df = edges_df.reset_index()

    print('generating network with pandana...')
    network = pandana.Network(nodes_df['x'], nodes_df['y'],
                   edges_df['u'], edges_df['v'],
                   edges_df[['length']])


    network.precompute(cutoff + 1)

    # first download the points of interest corresponding to the specified amenity type
    network.set_pois(category=POIt1, maxdist=cutoff, maxitems=N_pois, x_col=pois['lon'], y_col=pois['lat'])
    result = network.nearest_pois(distance=cutoff, category=POIt1, num_pois=N_pois)

    print('network generated inside callback')
    d = {'Lat': network.nodes_df.y, 'Lon': network.nodes_df.x, 'Distance': result[1]}
    df = pd.DataFrame(data=d,columns=['Lat', 'Lon', 'Distance'])

    fig = px.scatter_mapbox(df,
                        lat=df['Lat'],
                        lon=df['Lon'],
                        color=df['Distance'],
                        color_continuous_scale=['Green','Orange','Red'],
                        mapbox_style = 'open-street-map',
                        size_max=10, opacity = 0.5,
                        zoom=14)

    #                  color_continuous_scale='Bluered',
    #                  color_continuous_scale='Viridis',
    fig.update_traces(marker={'size': 10})
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    try:
     os.system("rm -r /home/beppe23/cache")
     os.system("rm -rf ~/.cache")
    except:
      pass
    #return [map_children, True]
    #return json.dumps(bbox)
    return [fig, True]
    #return clickD




if __name__ == "__main__":
    #app.run_server(port=8052, Debug=True)
    app.run_server(port=8052)
    #print(eval_js("google.colab.kernel.proxyPort(8058)"))
    #!python -m http.server 8058
    #app.run_server(mode='external')
    #app.run_server(mode='inline', Debug=True)

