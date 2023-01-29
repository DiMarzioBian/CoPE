MAPPING_DATASET = {
    'mtv': ['movietv', 'Movies_and_TV_5.json'],
    'e': ['electronics', 'Electronics_5.json'],
    'tool': ['tools', 'Tools_and_Home_Improvement_5.json'],
    'game': ['videogames', 'Video_Games_5.json'],
    'video': ['aivideo', 'Amazon_Instant_Video_5.json'],
    'garden': ['garden', 'Patio_Lawn_and_Garden_5.json'],
    'ml': ['ml-100k', 'u.data'],
    'mlm': ['ml-1m', 'ratings.dat'],
    'yoo': ['yoochoosebuy_cope', 'yoochoose-buys.dat']
}

MAPPING_TS_UNIT = {
    'movietv': 24 * 60 * 60,
    'electronics': 24 * 60 * 60,
    'videogames': 24 * 60 * 60,
    'tools': 24 * 60 * 60,
    'aivideo': 24 * 60 * 60,
    'garden': 24 * 60 * 60,
    'ml-100k': 24 * 60 * 60,
    'ml-1m': 24 * 60 * 60,
    'yoochoosebuy_cope': 24 * 60 * 60
}

MAPPING_META = {
    'videogames': 'meta_Video_Games.json'
}

IDX_PAD = 0

U_MARK_RAW = ['A2UAFI1CE7PDT8', 'AW3VZ5O895LRK', 'A2WVMOZ9XUGK4I']
# U_MARK_RAW = ['A2UAFI1CE7PDT8', 'A2WVMOZ9XUGK4I']
I_MARK_RAW = ['B006Z48TZS', 'B00APE00H4', 'B00H7NDSPC']

U_MARK = [406, 212, 1377]
# U_MARK = [406, 1377]
I_MARK = [362, 771, 1467]

OCCUR_U_MARK = [10, 92, 11]
