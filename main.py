import ipyleaflet
import os
from ipywidgets import Layout

defaultLayout = Layout(width='50%', height='400px')  # Layout карты (для правильного отображения)

Map = ipyleaflet.Map(zoom=6, layout=defaultLayout)  # Создание карты

os.chdir(os.path.dirname(os.path.realpath(__file__)))  # Установка рабочей директории


def getmap(name: str = 'map') -> str:
    html_file = os.path.join(os.getcwd(), 'static', f'{name}.html')
    Map.save(html_file)
    return name


if __name__ == '__main__':
    getmap()
