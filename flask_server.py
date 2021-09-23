import folium, ee
from flask import Flask
from map import add_ee_layer_to_map
from basemaps import basemaps

app = Flask(__name__)

folium.Map.add_ee_layer_to_map = add_ee_layer_to_map

@app.route('/')
def index():
    map = folium.Map(location=(55.74583, 37.61933))
    img_coll = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                    .filterDate('2013-01-01', '2018-12-31')
                    .filterBounds(ee.Geometry.Point(coords=[37.61933, 55.74583]))
                    .filter(ee.Filter.lt('CLOUD_COVER', 50)))
    landTrendr = ee.Algorithms.TemporalSegmentation.LandTrendr(img_coll, 100)

    basemaps['Google Maps'].add_to(map)
    basemaps['Google Satellite Hybrid'].add_to(map)
    
    map.add_ee_layer_to_map(img_coll, name='Landsat', visual_params={'bands':['SR_B3', 'SR_B2', 'SR_B1']})
    map.add_ee_layer_to_map(landTrendr, name='LandTrendr')
    map.add_child(folium.LayerControl())
    return map._repr_html_()

if __name__ == '__main__':
    ee.Initialize()
    app.run(debug=True)