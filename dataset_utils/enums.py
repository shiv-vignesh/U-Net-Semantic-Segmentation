
class Label:

    def __init__(self, label_tuple:tuple):
        
        self.label_name = label_tuple[0]
        self.id = label_tuple[1]
        self.category_name = label_tuple[3]
        self.category_id = label_tuple[4]
        self.color = label_tuple[-1]        

    def __str__(self):
        return f'{self.label_name} - {self.id} - {self.category_id} - {self.category_name} - {self.color}'

TRAIN_CITIES = ['monchengladbach', 'krefeld', 'cologne', 'jena', 'bochum', 'hamburg', 'dusseldorf', 'ulm', 'bremen', 'stuttgart', 'strasbourg', 'weimar', 'erfurt', 'darmstadt', 'zurich', 'tubingen', 'aachen', 'hanover']
TEST_CITIES = ['bonn', 'munich', 'bielefeld', 'berlin', 'leverkusen', 'mainz']
VAL_CITIES = ['frankfurt', 'munster', 'lindau']

VOID_CLASSES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]

# these are 19
VALID_CLASSES = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]

COLOR_DICT = {
    "Road (ID: 7)" : (128, 64, 128),
    "Sidewalk (ID: 8)" : (244, 35, 232),
    "Building (ID: 11)" : (70, 70, 70),
    "Wall (ID: 12)" : (102, 102, 156),
    "Fence (ID: 13)" : (190, 153, 153),
    "Pole (ID: 17)" : (153, 153, 153),
    "Traffic Light (ID: 19)" : (250, 170, 30),
    "Traffic Sign (ID: 20)" : (220, 220, 0),
    "Vegetation (ID: 21)" : (107, 142, 35),
    "Terrain (ID: 22)" : (152, 251, 152),
    "Sky (ID: 23)" : (70, 130, 180),
    "Person (ID: 24)" : (220, 20, 60),
    "Rider (ID: 25)" : (255, 0, 0),
    "Car (ID: 26)" : (0, 0, 142),
    "Truck (ID: 27)" : (0, 0, 70),
    "Bus (ID: 28)" : (0, 60, 100),
    "Train (ID: 31)" : (0, 80, 100),
    "Motorcycle (ID: 32)" : (0, 0, 230),
    "Bicycle (ID: 33)" : (119, 11, 32)
}

CLASS_ID_TO_COLOR_CODE = {
    7 : (128, 64, 128),
    8 : (244, 35, 232),
    11 : (70, 70, 70),
    12 : (102, 102, 156),
    13 : (190, 153, 153),
    17 : (153, 153, 153),
    19 : (250, 170, 30),
    20 : (220, 220, 0),
    21 : (107, 142, 35),
    22 : (152, 251, 152),
    23 : (70, 130, 180),
    24 : (220, 20, 60),
    25 : (255, 0, 0),
    26 : (0, 0, 142),
    27 : (0, 0, 70),
    28 : (0, 60, 100),
    31 : (0, 80, 100),
    32 : (0, 0, 230),
    33 : (119, 11, 32)
}

LABELS_LIST = [('unlabeled',0 ,255 , 'void', 0 , False, True, (0,0,0) ),
 ('ego vehicle',1 ,255 , 'void', 0 , False, True, (0,0,0) ),
 ('rectification border' ,2 ,255 , 'void', 0 , False, True, (0,0,0) ),
 ('out of roi' ,3 ,255 , 'void', 0 , False, True, (0,0,0) ),
 ('static',4 ,255 , 'void', 0 , False, True, (0,0,0) ),
 ('dynamic',5 ,255 , 'void', 0 , False, True, (111, 74,0) ),
 ('ground',6 ,255 , 'void', 0 , False, True, ( 81,0, 81) ),
 ('road',7 ,0 , 'flat', 1 , False, False, (128, 64,128) ),
 ('sidewalk' ,8 ,1 , 'flat', 1 , False, False, (244, 35,232) ),
 ('parking',9 ,255 , 'flat', 1 , False, True, (250,170,160) ),
 ('rail track' , 10 ,255 , 'flat', 1 , False, True, (230,150,140) ),
 ('building' , 11 ,2 , 'construction' , 2 , False, False, ( 70, 70, 70) ),
 ('wall', 12 ,3 , 'construction' , 2 , False, False, (102,102,156) ),
 ('fence' , 13 ,4 , 'construction' , 2 , False, False, (190,153,153) ),
 ('guard rail' , 14 ,255 , 'construction' , 2 , False, True, (180,165,180) ),
 ('bridge', 15 ,255 , 'construction' , 2 , False, True, (150,100,100) ),
 ('tunnel', 16 ,255 , 'construction' , 2 , False, True, (150,120, 90) ),
 ('pole', 17 ,5 , 'object', 3 , False, False, (153,153,153) ),
 ('polegroup', 18 ,255 , 'object', 3 , False, True, (153,153,153) ),
 ('traffic light', 19 ,6 , 'object', 3 , False, False, (250,170, 30) ),
 ('traffic sign', 20 ,7 , 'object', 3 , False, False, (220,220,0) ),
 ('vegetation' , 21 ,8 , 'nature', 4 , False, False, (107,142, 35) ),
 ('terrain', 22 ,9 , 'nature', 4 , False, False, (152,251,152) ),
 ('sky', 23 , 10 , 'sky' , 5 , False, False, ( 70,130,180) ),
 ('person', 24 , 11 , 'human' , 6 , True, False, (220, 20, 60) ),
 ('rider' , 25 , 12 , 'human' , 6 , True, False, (255,0,0) ),
 ('car', 26 , 13 , 'vehicle', 7 , True, False, (0,0,142) ),
 ('truck' , 27 , 14 , 'vehicle', 7 , True, False, (0,0, 70) ),
 ('bus', 28 , 15 , 'vehicle', 7 , True, False, (0, 60,100) ),
 ('caravan', 29 ,255 , 'vehicle', 7 , True, True, (0,0, 90) ),
 ('trailer', 30 ,255 , 'vehicle', 7 , True, True, (0,0,110) ),
 ('train' , 31 , 16 , 'vehicle', 7 , True, False, (0, 80,100) ),
 ('motorcycle' , 32 , 17 , 'vehicle', 7 , True, False, (0,0,230) ),
 ('bicycle', 33 , 18 , 'vehicle', 7 , True, False, (119, 11, 32) ),
 ('license plate', -1 , -1 , 'vehicle', 7 , False, True, (0,0,142) )]

VISUALIZATION_FILES = ['munster_000029_000019',
                    'munster_000101_000019',
                    'munster_000029_000019',
                    'munster_000101_000019',
                    'munster_000143_000019',
                    'munster_000014_000019',
                    'munster_000009_000019',
                    'munster_000009_000019',
                    'munster_000014_000019',
                    'munster_000143_000019',
                    'frankfurt_000001_068063',
                    'frankfurt_000001_068063',
                    'frankfurt_000001_056580',
                    'frankfurt_000001_056580',
                    'lindau_000031_000019',
                    'lindau_000031_000019']

''' 
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

'''