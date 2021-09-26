import folium, ee, LandTrendr, pprint
from flask import Flask
from map import add_ee_layer_to_map
from basemaps import basemaps

app = Flask(__name__)

folium.Map.add_ee_layer_to_map = add_ee_layer_to_map

@app.route('/')
def index():
    map = folium.Map(location=(55.74583, 37.61933), zoom_start=14)

    start_year = 1985
    end_year = 2017
    start_day = '06-20'
    end_day = '09-20'
    aoi = ee.Geometry.Point(37.61933, 55.74583)
    index = 'NBR'
    mask_these = ['cloud', 'shadow', 'snow', 'water']

    runParams = { 
        'maxSegments':            6,
        'spikeThreshold':         0.9,
        'vertexCountOvershoot':   3,
        'preventOneYearRecovery': True,
        'recoveryThreshold':      0.25,
        'pvalThreshold':          0.05,
        'bestModelProportion':    0.75,
        'minObservationsNeeded':  6
    }

    changeParams = {
        'delta':  'all',
        'sort':   'greatest',
        'year':   {'checked':True, 'start':1986, 'end':2017},
        'mag':    {'checked':True, 'value':200,  'operator':'>', 'dnsr': False},
        'dur':    {'checked':True, 'value':4,    'operator':'<'},
        'preval': {'checked':True, 'value':300,  'operator':'>'},
        'mmu':    {'checked':True, 'value':11},
        'index': index
    }

    lt = LandTrendr.runLT(start_year, end_year, start_day, end_day, aoi, index, [], runParams, mask_these)

    change_img = LandTrendr.getChangeMap(lt, changeParams)

    palette = ['#9400D3', '#4B0082', '#0000FF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000']

    yodVizParms = {
        'min': start_year,
        'max': end_year,
        'palette': palette,
    }

    magVizParms = {
        'min': 200,
        'max': 800,
        'palette': palette,
    }

    # basemaps['Google Maps'].add_to(map)
    # basemaps['Google Satellite Hybrid'].add_to(map)
    

    #map.add_ee_layer_to_map(change_img.select(['yod']),'Year of Detection', yodVizParms)
    map.add_ee_layer_to_map(change_img.select(['mag']), 'Magnitude of Change', magVizParms)

    map.add_child(folium.LayerControl())
    return map._repr_html_()

if __name__ == '__main__':
    ee.Initialize()
    app.run(debug=True)