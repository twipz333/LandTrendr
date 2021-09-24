from ee.batch import Export
import ipyleaflet, os, ee
from ipywidgets import Layout


defaultLayout = Layout(width='50%', height='400px')  # Layout карты (для правильного отображения)

Map = ipyleaflet.Map(zoom=6, layout=defaultLayout)  # Создание карты

os.chdir(os.path.dirname(os.path.realpath(__file__)))  # Установка рабочей директории

ee.Initialize()

def getmap() -> ipyleaflet.Map:
    if not Map:
        raise Exception('Map not created yet!')
    else:
        return Map

image_collection = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                        .filterDate('2013-01-01', '2018-12-31'))
                        
landTrendr = ee.Algorithms.TemporalSegmentation.LandTrendr(image_collection, 100)

Export.image.toDrive(image=landTrendr,
                        description='LandTrendr result',
                        folder=os.getcwd(), fileNamePrefix='landTrendr', fileFormat='png')

print(os.getcwd())

image = ipyleaflet.ImageOverlay()

def map_to_html(name: str = 'map') -> str:
    html_file = os.path.join(os.getcwd(), 'static', f'{name}.html')
    Map.save(html_file)
    return name

# if __name__ == '__main__':
#     map_to_html()
