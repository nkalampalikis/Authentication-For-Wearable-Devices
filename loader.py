from db import DB
import os.path


def load_db(window_size):

    if not os.path.isdir("./databases_BVP/"):
        print("Creating Databases Directory")
        os.mkdir("./databases_BVP/")
    # Samples to load
    db = DB("./databases_BVP/db_" + str(window_size) + ".sqlite", init=True)

    targets = list(range(1, 29))

    # Window Size
    for t in [window_size]:
        # Participant
        for i in targets:
            # Session
            for j in range(1, 5):
                # Sequence
                for k in range(1, 6):
                    g_file = "./data/{}/{}/{}/gyro.csv".format(i, j, k)
                    a_file = "./data/{}/{}/{}/accel.csv".format(i, j, k)

                    if os.path.isfile(g_file) and os.path.isfile(a_file):
                        print("Loading %d-%d-%d" % (i, j, k))
                        db.load_from_csv(g_file, a_file, str(i), j, k, t, 50)
                    else:
                        # print("Skipping %d-%d-%d (does not exist)" % (i,j,k))
                        pass


load_db(2)
load_db(3)
load_db(4)
load_db(5)
