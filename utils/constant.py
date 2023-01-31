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

I_MARK_RAW = ['B006Z48TZS', 'B00APE00H4', 'B00H7NDSPC']
I_MARK = [362, 771, 1467]


# # Justified loyal user
# U_MARK_RAW = ['A2UAFI1CE7PDT8', 'AW3VZ5O895LRK', 'A2WVMOZ9XUGK4I']
# U_MARK = [406, 212, 1377]
# OCCUR_U_MARK = [10, 92, 11]

# # New Justified user reacted s4 during burst
# U_MARK_RAW = ['AG4YSMAQGMRI7', 'A2Z3B0N8VDIZ78']
# U_MARK = [1577, 2934]
# OCCUR_U_MARK = [7, 5]

# # New Justified user reacted s4 during burst _ v4
U_MARK_RAW = ['A1EJT7NW91NWMP', 'ANBYFS0SVUZO2', 'A3POM5GPI6OW9W', 'A212DIOR7TFDHC']
U_MARK = [1385, 1741, 1747, 2046]
OCCUR_U_MARK = [23, 6, 24, 9]
