import folium, ee

def add_ee_layer_to_map(self, ee_object, name, visual_params=None):
    try:
        # ee.Image()
        if isinstance(ee_object, ee.image.Image):
            map_id = ee.Image(ee_object).getMapId(vis_params=visual_params)
            folium.raster_layers.TileLayer(
                tiles = map_id['tile_fetcher'].url_format,
                attr = 'Google Earth Engine',
                name = name,
                overlay = True,
                control = True
            ).add_to(self)
        
        # ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            ee_obj_new = ee_object.mosaic()
            map_id = ee.Image(ee_obj_new).getMapId(vis_params=visual_params)
            folium.raster_layers.TileLayer(
                tiles = map_id['tile_fetcher'].url_format,
                attr = 'Google Earth Engine',
                name = name,
                overlay = True,
                control = True
            ).add_to(self)

        # ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):    
            folium.GeoJson(
                data = ee_object.getInfo(),
                name = name,
                overlay = True,
                control = True
            ).add_to(self)
        
        # ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
            ee_obj_new = ee.Image().paint(ee_object, 0, 2)
            map_id = ee.Image(ee_obj_new).getMapId(vis_params=visual_params)
            folium.raster_layers.TileLayer(
                tiles = map_id['tile_fetcher'].url_format,
                attr = 'Google Earth Engine',
                name = name,
                overlay = True,
                control = True
            ).add_to(self)

    except Exception as ex:
        print(f'Невозможно отобразить {name}:{ex}')
