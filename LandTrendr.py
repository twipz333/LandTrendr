from json import dump
from math import pi
from typing import Dict, List, Literal, Union
import ee, datetime

from ee.image import Image
from ee.imagecollection import ImageCollection

def getChangeMap(lt: Union[Image, List[Image], ImageCollection], changeParams: Dict) -> ee.Image:
    """[summary]

    Args:
        lt (ee.ImageCollection): LandTrendr result
        changeParams (Dict):
    {
    `delta`:  'Loss',
    `sort`:   'Greatest',
    `year`:   {checked:false, start:1984, end:2017},
    `mag`:    {checked:true,  value:200,  operator: '>', dsnr:false},
    `dur`:    {checked:true,  value:4,    operator: '<'},
    `preval`: {checked:true,  value:300,  operator: '>'},
    `mmu`:    {checked:true,  value:11}
    }
    """
    # get the segment info
    segInfo = getSegmentData(lt, changeParams['index'], changeParams['delta'])

    changeMask = segInfo.arraySlice(0, 4, 5).gt(0)
    segInfo = segInfo.arrayMask(changeMask)
    
    #filter by year
    if makeBoolean(changeParams['year']['checked']) == True:
        yodArr = segInfo.arraySlice(0, 0, 1).add(1)
        yearMask = yodArr.gte(changeParams['year']['start']).And(yodArr.lte(changeParams['year']['end']))
        segInfo = segInfo.arrayMask(yearMask)
    

    #filter by mag
    magBand = {'axis': 0, 'start': 4, 'end': 5}
    if makeBoolean(changeParams['mag']['checked']) == True:
        if makeBoolean(changeParams['mag']['dnsr']) == True:
            magBand = {'axis': 0, 'start': 7, 'end': None} 
        if changeParams['mag']['operator'] == '<':
            magMask = segInfo.arraySlice(**magBand).lt(changeParams['mag']['value'])
        elif changeParams['mag']['operator'] == '>':
            magMask = segInfo.arraySlice(**magBand).gt(changeParams['mag']['value'])
        else:
            print("Error: provided mag operator does match either '>' or '<'")
        
        segInfo = segInfo.arrayMask(magMask)
    
    # filter by dur
    durBand = {'axis': 0, 'start': 5, 'end': 6}
    if makeBoolean(changeParams['dur']['checked']) == True:
        if changeParams['dur']['operator'] ==  '<':
            durMask = segInfo.arraySlice(**durBand).lt(changeParams['dur']['value'])
        elif changeParams['mag']['operator'] == '>':
            durMask = segInfo.arraySlice(**durBand).gt(changeParams['dur']['value'])
        else:
            print("Error: provided dur operator does match either '>' or '<'")
        segInfo =segInfo.arrayMask(durMask)

    # filter by preval
    prevalBand = {'axis': 0, 'start': 2, 'end': 3}
    if makeBoolean(changeParams['preval']['checked']) == True:
        if changeParams['preval']['operator'] == '<':
            prevalMask = segInfo.arraySlice(**prevalBand).lt(changeParams['preval']['value'])
        elif changeParams['preval']['operator'] == '>':
            prevalMask = segInfo.arraySlice(**prevalBand).gt(changeParams['preval']['value'])
        else:
            print("Error: provided preval operator does match either '>' or '<'")
        segInfo = segInfo.arrayMask(prevalMask)

    # sort by dist type
    sort = changeParams['sort'].lower()
    if sort == 'greatest':
        sortByThis = segInfo.arraySlice(0,4,5).multiply(-1) # need to flip the delta here, since arraySort is working by ascending order
    elif sort == 'least':
        sortByThis = segInfo.arraySlice(0,4,5)
    elif sort == 'newest':
        sortByThis = segInfo.arraySlice(0,0,1).multiply(-1) # need to flip the delta here, since arraySort is working by ascending order
    elif sort == 'oldest':
        sortByThis = segInfo.arraySlice(0,0,1)
    elif sort == 'fastest':
        sortByThis = segInfo.arraySlice(0,5,6)
    elif sort == 'slowest':
        sortByThis = segInfo.arraySlice(0,5,6).multiply(-1) # need to flip the delta here, since arraySort is working by ascending order

    segInfoSorted = segInfo.arraySort(sortByThis) # sort the array by magnitude

    chngArray = segInfoSorted.arraySlice(1, 0, 1) # get the first

    #make an image from the array of attributes for the change of interest
    arrRowNames = [['startYear', 'endYear', 'preval', 'postval', 'mag', 'dur', 'rate', 'csnr']]
    chngImg = chngArray.arrayProject([0]).arrayFlatten(arrRowNames)
    yod = chngImg.select('startYear').add(1).toInt16().rename('yod')
    chngImg = chngImg.addBands(yod).select(['yod', 'mag', 'dur', 'preval', 'rate', 'csnr'])

    # Mask for change/no change
    chngImg = chngImg.updateMask(chngImg.select('mag').gt(0))

    # Filter by MMU on year of change detection
    if makeBoolean(changeParams['mmu']['checked']) == True:
        if changeParams['mmu']['value'] > 1:
            mmuMask = (chngImg.select(['yod'])
                        .connectedPixelCount(changeParams['mmu']['value'], True)
                        .gte(changeParams['mmu']['value']))
            chngImg = chngImg.updateMask(mmuMask)

    return chngImg
        

def getSegmentData(lt: List[ee.Image], index: str, delta: str, right: bool=False) -> ee.Image:

    ltlt = lt.select('LandTrendr')           # select the LandTrendr band
    print(ltlt)
    rmse = lt.select('rmse')                  # select the rmse band
    vertexMask = ltlt.arraySlice(0, 3, 4)     # slice out the 'Is Vertex' row - yes(1)/no(0)
    vertices = ltlt.arrayMask(vertexMask)     # use the 'Is Vertex' row as a mask for all rows
    leftList = vertices.arraySlice(1, 0, -1)    # slice out the vertices as the start of segments
    rightList = vertices.arraySlice(1, 1) # slice out the vertices as the end of segments
    startYear = leftList.arraySlice(0, 0, 1)    # get year dimension of LT data from the segment start vertices
    startVal = leftList.arraySlice(0, 2, 3)     # get spectral index dimension of LT data from the segment start vertices
    endYear = rightList.arraySlice(0, 0, 1)     # get year dimension of LT data from the segment end vertices 
    endVal = rightList.arraySlice(0, 2, 3)      # get spectral index dimension of LT data from the segment end vertices
    dur = endYear.subtract(startYear)       # subtract the segment start year from the segment end year to calculate the duration of segments 
    mag = endVal.subtract(startVal)         # substract the segment start index value from the segment end index value to calculate the delta of segments
    rate = mag.divide(dur)                  # calculate the rate of spectral change    
    dsnr = mag.divide(rmse)             # make mag relative to fit rmse

    # whether to return all segments or either dist or grow

    if delta.lower() == 'all':
    # if the data should be set to the correct orientation, adjust it - don't need to do this for either gain or loss mag/rate/dsnr because it is set to absolute
        if right == True:
            if indexFlipper(index) == -1:
                startVal = startVal.multiply(-1)
                endVal = endVal.multiply(-1)
                mag = mag.multiply(-1)
                rate = rate.multiply(-1)
                dsnr = dsnr.multiply(-1)
      
    # now just get out - return the result 
        return (ee.Image.cat(([startYear, endYear, startVal, endVal, mag, dur, rate, dsnr]))
                    .unmask(ee.Image([[-9999]]))
                    .toArray(0))
    elif delta.lower() == 'gain' or delta.lower() == 'loss':
        if delta.lower() == 'gain':
            changeTypeMask = mag.lt(0)
        elif delta.lower() == 'loss':
            changeTypeMask = mag.gt(0)
  
        flip = indexFlipper(index)  
        
        return (ee.Image.cat([
            startYear.arrayMask(changeTypeMask),#.unmask(ee.Image(ee.Array([[-9999]]))),
            endYear.arrayMask(changeTypeMask),#.unmask(ee.Image(ee.Array([[-9999]]))),
            startVal.arrayMask(changeTypeMask).multiply(flip),#.unmask(ee.Image(ee.Array([[-9999]]))),
            endVal.arrayMask(changeTypeMask).multiply(flip),#.unmask(ee.Image(ee.Array([[-9999]]))),
            mag.arrayMask(changeTypeMask).abs(),#.unmask(ee.Image(ee.Array([[-9999]]))),
            dur.arrayMask(changeTypeMask),#.unmask(ee.Image(ee.Array([[-9999]]))),
            rate.arrayMask(changeTypeMask).abs(),#.unmask(ee.Image(ee.Array([[-9999]]))),
            dsnr.arrayMask(changeTypeMask).abs(),#.unmask(ee.Image(ee.Array([[-9999]])))
        ])
        .unmask(ee.Image([[-9999]]))
        .toArray(0))
    
def indexFlipper(index: str) -> int:
    indexObj = {
        'NBR': -1,
        'NDVI': -1,
        'NDSI': -1,  # ???? this is a tricky one
        'NDMI': -1,
        'EVI': -1,  
        'TCB': 1,
        'TCG': -1,
        'TCW': -1,
        'TCA': -1,
        'B1': 1,
        'B2': 1,
        'B3': 1,
        'B4': -1,
        'B5': 1,
        'B7': 1,
        'ENC': 1,
        'ENC1': 1,
        'TCC': 1,  
        'NBRz': 1,
        'B5z': 1
    }
    return indexObj[index]

def harmonizationRoy(oli: ee.Image) -> Image:
    slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949])        # RMA - create an image of slopes per band for L8 TO L7 regression line - David Roy
    itcp = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029])     # RMA - create an image of y-intercepts per band for L8 TO L7 regression line - David Roy
    # slopes = ee.Image.constant([0.885, 0.9317, 0.9372, 0.8339, 0.8639, 0.9165])       # least squares OLI to ETM+
    # itcp = ee.Image.constant([0.0183, 0.0123, 0.0123, 0.0448, 0.0306, 0.0116])        # least squares OLI to ETM+
    y = (oli.select(['B2','B3','B4','B5','B6','B7'],['B1', 'B2', 'B3', 'B4', 'B5', 'B7']) # select OLI bands 2-7 and rename them to match L7 band names
            .resample('bicubic')                                                          # ...resample the L8 bands using bicubic
            .subtract(itcp.multiply(10000)).divide(slopes)                                # ...multiply the y-intercept bands by 10000 to match the scale of the L7 bands then apply the line equation - subtract the intercept and divide by the slope
            .set('system:time_start', oli.get('system:time_start')))                      # ...set the output system:time_start metadata to the input image time_start otherwise it is null
    return y.toShort()                                                                       # return the image as short to match the type of the other data

def filterCollection(year: int, startDay: str, endDay: str, sensor: str, aoi: ee.Geometry) -> ee.ImageCollection:
    return (ee.ImageCollection('LANDSAT/'+ sensor + '/C01/T1_SR')
        .filterBounds(aoi)
        .filterDate(f"{year}-{startDay}", f"{year}-{endDay}"))


def removeImages(collection: ImageCollection, exclude: Dict) -> ImageCollection:
    #["LANDSAT/LC08/C01/T1_SR/LC08_046028_20170815"](system:id) or [LC08_046028_20170815](system:index)
    #could not get (system:id) to work though, so getting via string split and slice
    if exclude.get('exclude'):
        exclude = exclude['exclude']
        if exclude.get('imgIds'):
            excludeList = exclude['imgIds']
            for i in range(len(excludeList)):
                collection = collection.filter(ee.Filter.neq('system:index', excludeList[i].split('/').slice(-1).toString())) #system:id
        if exclude.get('slcOff'):
            if exclude['slcOff'] == True:
                collection = collection.filter(ee.Filter.And(ee.Filter.eq('SATELLITE', 'LANDSAT_7'), ee.Filter.gt('SENSING_TIME', '2003-06-01T00:00')).Not())

    return collection


def buildSensorYearCollection(year: int, startDay: str, endDay: str, sensor: str, aoi: ee.Geometry, exclude: Dict) -> ee.ImageCollection:
    startMonth = int(startDay[:2])
    endMonth = int(endDay[:2])
    if startMonth > endMonth:
        oldYear = str(int(year)-1)
        newYear = year
        oldYearStartDay = startDay
        oldYearEndDay = '12-31'
        newYearStartDay = '01-01'
        newYearEndDay = endDay
        
        oldYearCollection = filterCollection(oldYear, oldYearStartDay, oldYearEndDay, sensor, aoi)
        newYearCollection = filterCollection(newYear, newYearStartDay, newYearEndDay, sensor, aoi)
        
        srCollection = ee.ImageCollection(oldYearCollection.merge(newYearCollection))
    else:
        srCollection = filterCollection(year, startDay, endDay, sensor, aoi)
  
    srCollection = removeImages(srCollection, exclude)
  
    return srCollection


def getSRcollection(year: int, startDay: str, endDay: str, sensor: str,
        aoi: ee.Geometry, maskThese: List[str]=['cloud','shadow','snow'], exclude: Dict={}) -> Union[ee.ImageCollection, Literal['error']]:
    def mapping(img):
        dat = ee.Image(
            ee.Algorithms.If(
                sensor == 'LC08',                                                  # condition - if image is OLI
                harmonizationRoy(img.unmask()),                                    # true - then apply the L8 TO L7 alignment function after unmasking pixels that were previosuly masked (why/when are pixels masked)
                img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7'])                   # false - else select out the reflectance bands from the non-OLI image
           .unmask()                                                       # ...unmask any previously masked pixels 
           .resample('bicubic')                                            # ...resample by bicubic 
           .set('system:time_start', img.get('system:time_start'))         # ...set the output system:time_start metadata to the input image time_start otherwise it is null
        )
        )
    
        #    # make a cloud, cloud shadow, and snow mask from fmask band
        #     qa = img.select('pixel_qa')                                       # select out the fmask band
        #     mask = qa.bitwiseAnd(8).eq(0).and(                                 # include shadow
        #               qa.bitwiseAnd(16).eq(0)).and(                               # include snow
        #               qa.bitwiseAnd(32).eq(0))                                   # include clouds
    
        mask = ee.Image(1)
        if len(maskThese) != 0:
            qa = img.select('pixel_qa') 
            for i in maskThese:
                if i == 'water': mask = qa.bitwiseAnd(4).eq(0).multiply(mask)
                if i == 'shadow': mask = qa.bitwiseAnd(8).eq(0).multiply(mask)
                if i == 'snow': mask = qa.bitwiseAnd(16).eq(0).multiply(mask)
                if i == 'cloud': mask = qa.bitwiseAnd(32).eq(0).multiply(mask)

            return dat.mask(mask) #apply the mask - 0's in mask will be excluded from computation and set to opacity=0 in display
        
        else:
            return dat


    # make sure that mask labels are correct
    maskOptions = ['cloud', 'shadow', 'snow', 'water']
    for i in maskThese:
        i =  i.lower()
        test = maskOptions.index(i)
        if test == -1:
            print(f'error: {i} is not included in the list maskable features. Please see ___ for list of maskable features to include in the maskThese parameter')
            return 'error'

    
    # get a landsat collection for given year, day range, and sensor
    srCollection = buildSensorYearCollection(year, startDay, endDay, sensor, aoi, exclude)

    # apply the harmonization function to LC08 (if LC08), subset bands, unmask, and resample           
    srCollection = srCollection.map(mapping)

    return srCollection # return the prepared collection

def getCombinedSRcollection(year: int, startDay: str, endDay: str, aoi: ee.Geometry, maskThese: List[str], exclude={}) -> ee.ImageCollection:
    lt5 = getSRcollection(year, startDay, endDay, 'LT05', aoi, maskThese, exclude)       # get TM collection for a given year, date range, and area
    le7 = getSRcollection(year, startDay, endDay, 'LE07', aoi, maskThese, exclude)       # get ETM+ collection for a given year, date range, and area
    lc8 = getSRcollection(year, startDay, endDay, 'LC08', aoi, maskThese, exclude)       # get OLI collection for a given year, date range, and area
    mergedCollection = ee.ImageCollection(lt5.merge(le7).merge(lc8)) # merge the individual sensor collections into one imageCollection object
    #mergedCollection = removeImages(mergedCollection, exclude)
    return mergedCollection                                              # return the Imagecollection

def medoidMosaic(inCollection: ee.ImageCollection, dummyCollection: ee.ImageCollection) -> ee.ImageCollection:
    def mapping(img):
        diff = ee.Image(img).subtract(median).pow(ee.Image.constant(2))                                       # get the difference between each image/band and the corresponding band median and take to power of 2 to make negatives positive and make greater differences weight more
        return diff.reduce('sum').addBands(img)  
    # fill in missing years with the dummy collection
    imageCount = inCollection.toList(1).length()                                                            # get the number of images 
    finalCollection = ee.ImageCollection(ee.Algorithms.If(imageCount.gt(0), inCollection, dummyCollection)) # if the number of images in this year is 0, then use the dummy collection, otherwise use the SR collection
  
    # calculate median across images in collection per band
    median = finalCollection.median()                                                                       # calculate the median of the annual image collection - returns a single 6 band image - the collection median per band
  
    # calculate the different between the median and the observation per image per band
    difFromMedian = finalCollection.map(mapping)
  
    # get the medoid by selecting the image pixel with the smallest difference between median and observation per band 
    return ee.ImageCollection(difFromMedian).reduce(ee.Reducer.min(7)).select([1,2,3,4,5,6], ['B1','B2','B3','B4','B5','B7']) # find the powered difference that is the least - what image object is the closest to the median of teh collection - and then subset the SR bands and name them - leave behind the powered difference band


def buildMosaic(year: int, startDay: str, endDay: str, aoi: ee.Geometry, dummyCollection: ee.ImageCollection, maskThese: List[str], exclude={}) -> ee.Image:
    collection = getCombinedSRcollection(year, startDay, endDay, aoi, maskThese, exclude)  # get the SR collection
    img = (medoidMosaic(collection, dummyCollection)                     # apply the medoidMosaic function to reduce the collection to single image per year by medoid 
              .set('system:time_start', datetime.datetime(year, 8, 1).timestamp()))  # add the year to each medoid image - the data is hard-coded Aug 1st 
    return ee.Image(img)                                                   # return as image object


def buildSRcollection(startYear: int, endYear: int, startDay: str, endDay: str,
            aoi: ee.Geometry, maskThese: List[str]=['cloud', 'shadow', 'snow'], exclude={}) -> ee.ImageCollection:
    dummyCollection = ee.ImageCollection([ee.Image([0,0,0,0,0,0]).mask(ee.Image(0))]) # make an image collection from an image with 6 bands all set to 0 and then make them masked values
    imgs = []                                                                         # create empty array to fill
    for i in range(startYear, endYear + 1):                            # for each year from hard defined start to end build medoid composite and then add to empty img array
        tmp = buildMosaic(i, startDay, endDay, aoi, dummyCollection, maskThese, exclude)                    # build the medoid mosaic for a given year
        imgs.append(tmp.set('system:time_start', datetime.datetime(i,8,1).timestamp()))       # concatenate the annual image medoid to the collection (img) and set the date of the image - hard coded to the year that is being worked on for Aug 1st

    return ee.ImageCollection(imgs)

def nbrTransform(img: ee.Image) -> ee.Image:
    nbr = (img.normalizedDifference(['B4', 'B7']) # calculate normalized difference of B4 and B7. orig was flipped: ['B7', 'B4']
                 .multiply(1000) # scale results by 1000
                 .select([0], ['NBR']) # name the band
                 .set('system:time_start', img.get('system:time_start')))
    return nbr

def ndmiTransform(img: ee.Image) -> ee.Image:
    ndmi = (img.normalizedDifference(['B4', 'B5']) # calculate normalized difference of B4 and B7. orig was flipped: ['B7', 'B4']
                 .multiply(1000) # scale results by 1000
                 .select([0], ['NDMI']) # name the band
                 .set('system:time_start', img.get('system:time_start')))
    return ndmi

def ndviTransform(img: ee.Image) -> ee.Image:
    ndvi = (img.normalizedDifference(['B4', 'B3']) # calculate normalized dif between band 4 and band 3 (B4-B3/B4_B3)
                .multiply(1000) # scale results by 1000
                .select([0], ['NDVI']) # name the band
                .set('system:time_start', img.get('system:time_start')))
    return ndvi

def ndsiTransform(img: ee.Image) -> ee.Image:
    ndsi = (img.normalizedDifference(['B2', 'B5']) # calculate normalized dif between band 4 and band 3 (B4-B3/B4_B3)
                .multiply(1000) # scale results by 1000
                .select([0], ['NDSI']) # name the band
                .set('system:time_start', img.get('system:time_start')))
    return ndsi

def eviTransform(img: ee.Image) -> ee.Image:
    evi = (img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
        'NIR': img.select('B4'),
        'RED': img.select('B3'),
        'BLUE': img.select('B1')
    })
    .multiply(1000) # scale results by 1000
    .select([0], ['EVI']) # name the band
    .set('system:time_start', img.get('system:time_start'))) 
    return evi

def tcTransform(img: ee.Image) -> ee.Image:
    b = ee.Image(img).select(["B1", "B2", "B3", "B4", "B5", "B7"]) # select the image bands
    brt_coeffs = ee.Image.constant([0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303]) # set brt coeffs - make an image object from a list of values - each of list element represents a band
    grn_coeffs = ee.Image.constant([-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446]) # set grn coeffs - make an image object from a list of values - each of list element represents a band
    wet_coeffs = ee.Image.constant([0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109]) # set wet coeffs - make an image object from a list of values - each of list element represents a band

    sum = ee.Reducer.sum() # create a sum reducer to be applyed in the next steps of summing the TC-coef-weighted bands
    brightness = b.multiply(brt_coeffs).reduce(sum) # multiply the image bands by the brt coef and then sum the bands
    greenness = b.multiply(grn_coeffs).reduce(sum) # multiply the image bands by the grn coef and then sum the bands
    wetness = b.multiply(wet_coeffs).reduce(sum) # multiply the image bands by the wet coef and then sum the bands
    angle = (greenness.divide(brightness)).atan().multiply(180/pi).multiply(100)
    tc = (brightness.addBands(greenness)
                    .addBands(wetness)
                    .addBands(angle)
                    .select([0,1,2,3], ['TCB','TCG','TCW','TCA']) #stack TCG and TCW behind TCB with .addBands, use select() to name the bands
                    .set('system:time_start', img.get('system:time_start')))
    return tc

def calcIndex(img: ee.Image, index: str, flip: int) -> ee.Image:
# make sure index string in upper c.ase
    index = index.upper()
  
    # figure out if we need to calc tc
    tcList = ['TCB', 'TCG', 'TCW', 'TCA']
    if index in tcList: # Moddified
        tc = tcTransform(img)
    
    # need to flip some indices if this is intended for segmentation
    indexFlip = -1 if flip == 1 else 1
    
    # need to cast raw bands to float to make sure that we don't get errors regarding incompatible bands
    # ...derivations are already float because of division or multiplying by decimal
    if index == 'B1':
        indexImg = img.select(['B1']).float()#.multiply(indexFlip)   
    elif index == 'B2':
        indexImg = img.select(['B2']).float()#.multiply(indexFlip)    
    elif index == 'B3':
        indexImg = img.select(['B3']).float()#.multiply(indexFlip)    
    elif index == 'B4':
        indexImg = img.select(['B4']).multiply(indexFlip).float()    
    elif index == 'B5':
        indexImg = img.select(['B5']).float()#.multiply(indexFlip)    
    elif index == 'B7':
        indexImg = img.select(['B7']).float()#.multiply(indexFlip)   
    elif index == 'NBR':
        indexImg = nbrTransform(img).multiply(indexFlip)    
    elif index == 'NDMI':
        indexImg = ndmiTransform(img).multiply(indexFlip)    
    elif index == 'NDVI':
        indexImg = ndviTransform(img).multiply(indexFlip)    
    elif index == 'NDSI':
        indexImg = ndsiTransform(img).multiply(indexFlip)    
    elif index == 'EVI':
        indexImg = eviTransform(img).multiply(indexFlip)    
    elif index == 'TCB':
        indexImg = tc.select(['TCB'])#.multiply(indexFlip)    
    elif index == 'TCG':
        indexImg = tc.select(['TCG']).multiply(indexFlip)    
    elif index == 'TCW':
        indexImg = tc.select(['TCW']).multiply(indexFlip)    
    elif index == 'TCA':
        indexImg = tc.select(['TCA']).multiply(indexFlip)    
    else:
        print('The index you provided is not supported')


    return indexImg.set('system:time_start', img.get('system:time_start'))


def standardize(collection: ee.ImageCollection) -> ee.ImageCollection:
    mean = collection.reduce(ee.Reducer.mean())
    stdDev = collection.reduce(ee.Reducer.stdDev())

    meanAdj = collection.map(lambda img: img.subtract(mean).set('system:time_start', img.get('system:time_start')))

    return meanAdj.map(lambda img: img.divide(stdDev).set('system:time_start', img.get('system:time_start')))

def makeTCcomposite(annualSRcollection: ee.ImageCollection, reducer: str) -> ee.ImageCollection:
    def mapping_annual(img):
        tcb = calcIndex(img, 'TCB', 1)#.unmask(0)
        tcg = calcIndex(img, 'TCG', 1)#.unmask(0)
        tcw = calcIndex(img, 'TCW', 1)#.unmask(0)
        return (tcb.addBands(tcg)
                .addBands(tcw)
                .set('system:time_start', img.get('system:time_start')))

    def mapping_standart(img):
        imgCollection = ee.ImageCollection.fromImages([
                                            img.select(['TCB'],['Z']),
                                            img.select(['TCG'],['Z']),
                                            img.select(['TCW'],['Z'])])
        if reducer == 'mean':
            reducedImg = imgCollection.mean()
        elif reducer == 'max':
            reducedImg = imgCollection.max()
        elif reducer == 'sum':
            reducedImg = imgCollection.sum()
        else:
            print('The reducer you provided is not supported')

        return reducedImg.multiply(1000).set('system:time_start', img.get('system:time_start')) 
    
    TCcomposite = annualSRcollection.map(mapping_annual)

    tcb = TCcomposite.select(['TCB'])
    tcg = TCcomposite.select(['TCG'])
    tcw = TCcomposite.select(['TCW'])
    
    # standardize the TC bands
    tcbStandard = standardize(tcb)
    tcgStandard = standardize(tcg)
    tcwStandard = standardize(tcw)
  
    # combine the standardized TC band collections into a single collection
    tcStandard = tcbStandard.combine(tcgStandard).combine(tcwStandard)
  
    TCcomposite = tcStandard.map(mapping_standart)
    
    return TCcomposite


def makeEnsemblecomposite(annualSRcollection: ee.ImageCollection, reducer: str) -> ee.ImageCollection:
    def mapping_annual(img):
        b5   = calcIndex(img, 'B5', 1)
        b7   = calcIndex(img, 'B7', 1)
        tcw  = calcIndex(img, 'TCW', 1)
        tca  = calcIndex(img, 'TCA', 1)
        ndmi = calcIndex(img, 'NDMI', 1)
        nbr  = calcIndex(img, 'NBR', 1)
        return (b5.addBands(b7)
                .addBands(tcw)
                .addBands(tca)
                .addBands(ndmi)
                .addBands(nbr)
                .set('system:time_start', img.get('system:time_start')))
    
    def mapping_standart(img):
        imgCollection = ee.ImageCollection.fromImages([
            img.select(['B5'],['Z']),
            img.select(['B7'],['Z']),
            img.select(['TCW'],['Z']),
            img.select(['TCA'],['Z']),
            img.select(['NDMI'],['Z']),
            img.select(['NBR'],['Z']),])
        
        if reducer == 'mean':
            reducedImg = imgCollection.mean()
        elif reducer == 'max':
            reducedImg = imgCollection.max()
        elif reducer == 'sum':
            reducedImg = imgCollection.sum()
        else:
            print('The reducer you provided is not supported')   

        return reducedImg.multiply(1000).set('system:time_start', img.get('system:time_start'))
    
    
    # make a collection of the ensemble indices stacked as bands
    stack = annualSRcollection.map(mapping_annual)
  
    # make subset collections of each index
    b5 = stack.select('B5')
    b7 = stack.select('B7')
    tcw = stack.select('TCW')
    tca = stack.select('TCA')
    ndmi = stack.select('NDMI')
    nbr = stack.select('NBR')
        
    # standardize each index to mean 0 stdev 1
    b5Standard = standardize(b5)
    b7Standard = standardize(b7)
    tcwStandard = standardize(tcw)
    tcaStandard = standardize(tca)
    ndmiStandard = standardize(ndmi)
    nbrStandard = standardize(nbr)
        
    # combine the standardized band collections into a single collection
    standard = (b5Standard.combine(b7Standard).combine(tcwStandard).combine(tcaStandard)
                                .combine(ndmiStandard).combine(nbrStandard))
    
    # reduce the collection to a single value
    composite = standard.map(mapping_standart)      
    
    return composite

def makeEnsemblecomposite1(annualSRcollection: ee.ImageCollection, reducer: str) -> ee.ImageCollection:
    def mapping_annual(img):
        b5   = calcIndex(img, 'B5', 1)
        tcb  = calcIndex(img, 'TCB', 1)
        tcg  = calcIndex(img, 'TCG', 1)
        nbr  = calcIndex(img, 'NBR', 1)
        return (b5.addBands(tcb)
                .addBands(tcg)
                .addBands(nbr)
                .set('system:time_start', img.get('system:time_start')))
    
    def mapping_standart(img):
        imgCollection = ee.ImageCollection.fromImages([
            img.select(['B5'],['Z']),#.pow(ee.Image(1)).multiply(img.select('B5').gte(0).where(img.select('B5').lt(0),-1)),
            img.select(['TCB'],['Z']),#.pow(ee.Image(1.5)).multiply(img.select('TCB').gte(0).where(img.select('TCB').lt(0),-1)),
            img.select(['TCG'],['Z']),#.pow(ee.Image(1.5)).multiply(img.select('TCG').gte(0).where(img.select('TCG').lt(0),-1)),
            img.select(['NBR'],['Z'])])#.pow(ee.Image(1.5)).multiply(img.select('NBR').gte(0).where(img.select('NBR').lt(0),-1))

        if reducer == 'mean':
            reducedImg = imgCollection.mean()
        elif reducer == 'max':
            reducedImg = imgCollection.max()
        elif reducer == 'sum':
            reducedImg = imgCollection.sum()
        else:
            print('The reducer you provided is not supported')
        
        return reducedImg.multiply(1000).set('system:time_start', img.get('system:time_start'))
    
    # make a collection of the ensemble indices stacked as bands
    TCcomposite = annualSRcollection.map(mapping_annual)
    
    # make subset collections of each index
    b5 = TCcomposite.select('B5')
    tcb = TCcomposite.select('TCB')
    tcg = TCcomposite.select('TCG')
    nbr = TCcomposite.select('NBR')
        
    # standardize each index - get z-score
    b5Standard = standardize(b5)
    tcbStandard = standardize(tcb)
    tcgStandard = standardize(tcg)
    nbrStandard = standardize(nbr)
        
    # combine the standardized TC band collections into a single collection
    tcStandard = b5Standard.combine(tcbStandard).combine(tcgStandard).combine(nbrStandard)
    
    # reduce the collection to a single value
    TCcomposite = tcStandard.map(mapping_standart)  
    
    return TCcomposite

def standardizeIndex(collection: ee.ImageCollection, index: str) -> ee.ImageCollection:
    zCollection = collection.map(lambda img: calcIndex(img, index, 1))

    zCollection = standardize(zCollection)

    zCollection = zCollection.map(lambda img: img.multiply(1000).set('system:time_start', img.get('system:time_start')))

    return zCollection

def buildLTcollection(collection: ee.ImageCollection, index: str, ftvList: List[str]) -> ee.ImageCollection:
    def mapping(img):
        allStack = calcIndex(img, index, 1)
        for ftv in ftvList:
            ftvimg = (calcIndex(img, ftvList[ftv], 0)
                               .select([ftvList[ftv]],['ftv_'+ftvList[ftv].lower()]))
      
            allStack = (allStack.addBands(ftvimg)
                                        .set('system:time_start', img.get('system:time_start')))
        return allStack

    # tasseled cap composite
    if index == 'TCC':
      LTcollection = makeTCcomposite(collection, 'mean') 
    elif index == 'TCM':
      LTcollection = makeTCcomposite(collection, 'max') 
    elif index == 'TCS':
      LTcollection = makeTCcomposite(collection, 'sum') 
    
    # 6-band composite - Based on Figure 3 of the linked paper: https:#larse.forestry.oregonstate.edu/sites/larse/files/pub_pdfs/Cohen_et_al_2018.pdf
    elif index == 'ENC':
      LTcollection = makeEnsemblecomposite(collection, 'mean') 
    elif index == 'ENM':
      LTcollection = makeEnsemblecomposite(collection, 'max') 
    elif index == 'ENS':
      LTcollection = makeEnsemblecomposite(collection, 'sum') 
    
    # 6-band composite - Based on Table 5 of the linked paper: https:#larse.forestry.oregonstate.edu/sites/larse/files/pub_pdfs/Cohen_et_al_2018.pdf
    elif index == 'ENC1':
      LTcollection = makeEnsemblecomposite1(collection, 'mean') 
    elif index == 'ENM1':
      LTcollection = makeEnsemblecomposite1(collection, 'max') 
    elif index == 'ENS1':
      LTcollection = makeEnsemblecomposite1(collection, 'sum') 
    
    # standardized versions of indices: mean 0 stdDev 1  
    elif index == 'B5z':
      LTcollection = standardizeIndex(collection, 'B5') 
    elif index == 'B7z':
      LTcollection = standardizeIndex(collection, 'B7') 
    elif index == 'TCWz':
      LTcollection = standardizeIndex(collection, 'TCW') 
    elif index == 'TCAz':
      LTcollection = standardizeIndex(collection, 'TCA') 
    elif index == 'NDMIz':
      LTcollection = standardizeIndex(collection, 'NDMI') 
    elif index == 'NBRz':
      LTcollection = standardizeIndex(collection, 'NBR') 
      
    else:
      LTcollection = collection.map(mapping)

    return LTcollection


def runLT(startYear: int, endYear: int, startDay: str, endDay: str, aoi: ee.Geometry, index: str, ftvList: List[str],
            runParams: Dict, maskThese: List[str]=['cloud','shadow','snow'], exclude={}) -> ee.Image:
    annualSRcollection = buildSRcollection(startYear, endYear, startDay, endDay, aoi, maskThese, exclude) # collects surface reflectance images 
    annualLTcollection = buildLTcollection(annualSRcollection, index, ftvList)
    runParams['timeSeries'] = annualLTcollection
    return ee.Algorithms.TemporalSegmentation.LandTrendr(**runParams)

def makeBoolean(value) -> bool:
    if not isinstance(value, bool):
        try:
            value = value.lower() != 'false'
        except:
            pass

    return value