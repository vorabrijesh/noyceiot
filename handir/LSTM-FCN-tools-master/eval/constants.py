TRAIN_FILES = [
                '../data/ArrowHead_TRAIN',               # 0
                '../data/ArrowHead_EXP_TRAIN',           # 1
                '../data/Worms_TRAIN',                   # 2
                '../data/Worms_EXP_TRAIN',               # 3
                '../data/rsafr_TRAIN',#4
                '../data/rsafr_TRAIN',#5
                '../data/rsafr_TRAIN',#6
                '../data/CWEMalware_TRAIN',  # 7
                '../data/CWEMalware_TRAIN',  # 8
    '../data/60samples_TRAIN',  # 7
    '../data/60samples_TEST',  # 8


                ]


TEST_FILES = [
               '../data/ArrowHead_TEST',                # 0
               '../data/ArrowHead_TEST',                # 1
               '../data/Worms_TEST',                    # 2
               '../data/Worms_TEST',                    # 3
                '../data/rsafr_TEST',  # 4
                '../data/rsafr_EXP1_TEST',#5
                '../data/rsafr_EXP2_TEST',#6
                '../data/CWEMalware_TRAIN',  # 7
                '../data/CWEMalware_TEST',  # 8
    '../data/60samples_TRAIN',  # 7
    '../data/60samples_TRAIN',  # 8

              ]


MAX_SEQUENCE_LENGTH_LIST = [
                             251,  # 0
                             251,  # 1
                             900,  # 2
                             900,  # 3
                            399,#4
                            399,#5
                            399,#6
                            16,
                            16,
239,
239,
                            ]


NB_CLASSES_LIST = [
                    3,   # 0
                    3,   # 1
                    5,   # 2
                    5,   # 3
                    2, #4
                    2,#5
                    2,#6
                    2,
                    2,
    30,
    30,

                   ]


X_AXIS_LIST = [
                    'timestep',  # 0
                    'timestep',  # 1
                    'timestep',  # 2
                    'timestep',  # 3
                    'timestep',  # 4
                    'timestep',  # 5

                   ]


Y_AXIS_LIST = [
                    'value',  # 0
                    'value',  # 1
                    'value',  # 2
                    'value',  # 3
                    'value',  # 4
                    'value',  # 5

                   ]