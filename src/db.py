import sqlite3
import csv
import json
import numpy as np
from scipy.interpolate import interp1d
from .signal_filter import SignalFilter, SignalType

class DB:
    def __init__(self, db_file, init=False):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        if(init):
            data_table = self.cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?;
            """,  ["data"])

            # Iterate over all tables and drop them
            if(data_table.fetchone() != None):
                tables = self.cursor.execute("""
                    SELECT table_name FROM data
                """)
                for t in tables:
                    self.cursor.execute("DROP TABLE IF EXISTS `" + str(t[0]) + "`");

                self.cursor.execute("DROP TABLE IF EXISTS data");

            self.cursor.execute("""
            CREATE TABLE `data` (
                `subject_id`	TEXT NOT NULL,
                `session_id`    INTEGER NOT NULL,
                `sequence_id`   INTEGER NOT NULL,
                `window_sz`     INTEGER NOT NULL,
                `table_name`    TEXT NOT NULL,
                PRIMARY KEY(`subject_id`,`session_id`, `sequence_id`,`window_sz`,`table_name`)
            );
            """);

            self.conn.commit()
            print("Tables instantiated")

    @staticmethod
    def _align( start_time, data, hz ):
        # Figure out how many sample fall out of range and prune them
        end_time = data["TS"][-1]
        samples = 0
        for r in data["TS"]:
            if( r >= start_time ):
                break
            samples += 1
        if( samples > 0 ):
            data = np.split(data, [samples - 1])[1]


        # Check if TS has duplicates and remove if necessary
        s = np.sort(data["TS"], axis=None)
        kill_idexes = []
        for d in s[:-1][s[1:] == s[:-1]]:
            dups = np.where(data["TS"] == d)[0]
            for i in range(0, len(dups)-1):
                kill_idexes.append(dups[i])
        ts = np.delete(data["TS"], kill_idexes)
        x = np.delete(data["X"], kill_idexes)
        y = np.delete(data["Y"], kill_idexes)
        z = np.delete(data["Z"], kill_idexes)

        # Interpolate so samples are evenly spaced
        r = np.arange(start_time, end_time, int(1000 / hz))
        xinterp = (interp1d( ts, x ))(r)
        yinterp = (interp1d( ts, y ))(r)
        zinterp = (interp1d( ts, z ))(r)
        array = []
        for i in range(0, len(r)):
            TS = float( r[i] )
            X = float( xinterp[i] )
            Y = float( yinterp[i] )
            Z = float( zinterp[i] )
            array.append( ( TS, X, Y, Z ) )
        dtype = [
                    ('TS', 'float64'),
                    ('X', 'float64'),
                    ('Y', 'float64'),
                    ('Z', 'float64')
                ]
        data = np.array(array, dtype)
        return data

    @staticmethod
    def _synchronize(data_a, data_g, hz):
        # Identify common start point
        g_start = data_g[ "TS" ][ 0 ]
        a_start = data_a[ "TS" ][ 0 ]
        offset = g_start - a_start

        # gyro starts before accel
        if( offset < 0 ):
            #print( "=> Aligning" )
            data_g = DB._align( data_a["TS"][0], data_g, hz )
            data_a = DB._align( data_a["TS"][0], data_a, hz )
        # accel starts before gyro
        else:
            data_a = DB._align( data_g["TS"][0], data_a, hz )
            data_g = DB._align( data_g["TS"][0], data_g, hz )

        # Force length conformity
        g_len = len( data_g )
        a_len = len( data_a )
        diff = g_len - a_len

        # Gyro is longer
        if( g_len > a_len ):
            data_g = np.split( data_g, [ a_len ] )[ 0 ]
        # Accel is longer
        elif( a_len > g_len ):
            data_a = np.split( data_a, [ g_len ] )[ 0 ]
        return data_a, data_g

    def load_from_csv(self, gyro_file, accel_file, subject_id, session_id, sequence_id, window_sz, hz, signal_type=None):
        f_gyro = open(gyro_file, "r" )
        f_accel = open(accel_file, "r")
        reader_gyro = csv.DictReader( f_gyro )
        reader_accel = csv.DictReader( f_accel )

        def reader_to_array_bad(reader):
            output = []
            for row in reader:
                TS = int( row["TS"] )
                X = float( 0 )
                Y = float( 0 )
                Z = float( 0 )
                output.append( ( TS, X, Y, Z ) )


            dtype = [('TS', 'int64'), ('X', 'float64'), ('Y', 'float64'), ('Z', 'float64')]
            output = np.array(output, dtype)
            output = np.sort(output, order='TS', kind="mergesort")

            return output
        def reader_to_array(reader):
            output = []
            for row in reader:
                TS = int( row["TS"] )
                X = float( row["X"] )
                Y = float( row["Y"] )
                Z = float( row["Z"] )
                output.append( ( TS, X, Y, Z ) )


            dtype = [('TS', 'int64'), ('X', 'float64'), ('Y', 'float64'), ('Z', 'float64')]
            output = np.array(output, dtype)
            output = np.sort(output, order='TS', kind="mergesort")

            return output

        raw_gyro = reader_to_array(reader_gyro)
        raw_accel = reader_to_array(reader_accel)

        raw_accel, raw_gyro = DB._synchronize(raw_accel, raw_gyro, hz)

        # Creates overlaps in a normal array
        a = []
        g = []
        # Use overlapped data
        # r = list(range(0, window_sz*hz))
        r = list(range(0, hz))
        while len(raw_gyro) >= hz * window_sz and len(raw_accel) >= hz * window_sz:
            g.append(raw_gyro[:hz*window_sz])
            a.append(raw_accel[:hz*window_sz])
            raw_accel = np.delete(raw_accel,r,0)
            raw_gyro = np.delete(raw_gyro,r,0)

        # Merge accel and gyro segments into combined structured arrays
        max_seg = min(len(a), len(g))
        segments = []
        dtype = [
            ('TS', 'float64'),
            ('accel_X', 'float64'), ('accel_Y', 'float64'), ('accel_Z', 'float64'),
            ('gyro_X', 'float64'), ('gyro_Y', 'float64'), ('gyro_Z', 'float64')
        ]

        for i in range(max_seg):
            accel_seg, gyro_seg = a[i], g[i]
            if len(accel_seg) != len(gyro_seg):
                continue

            # Merge accel and gyro into single structured array
            merged = np.array([
                (
                    gyro_seg["TS"][j],
                    accel_seg["X"][j], accel_seg["Y"][j], accel_seg["Z"][j],
                    gyro_seg["X"][j], gyro_seg["Y"][j], gyro_seg["Z"][j]
                )
                for j in range(len(accel_seg))
            ], dtype=dtype)

            if len(merged) == window_sz * hz:
                segments.append(merged)

        # Default to BCG for backward compatibility
        if signal_type is None:
            signal_type = SignalType.BCG

        # Apply signal filter to each segment
        filtered_segments = []
        signal_name = signal_type.value.upper()
        print(f"\rCreating {signal_name} -/{len(segments)}", end='')

        for i, segment in enumerate(segments):
            print(f"\rCreating {signal_name} {i + 1}/{len(segments)}", end='')
            filtered = SignalFilter(segment, hz, signal_type)
            filtered_segments.append(filtered)
        print()


        # First, create a record for the new table
        table_name = "{}_{}_{}_{}".format(subject_id,session_id,sequence_id,window_sz)
        self.cursor.execute("""
                INSERT INTO data (subject_id,session_id,sequence_id,window_sz,table_name)
                VALUES (?,?,?,?,?);
            """,[subject_id,session_id,sequence_id,window_sz,table_name])

        # Create a new table for the db
        self.cursor.execute("CREATE TABLE `" + table_name +
        """` (
            `idex`	        INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            `segment_gyro_x`	TEXT NOT NULL,
            `segment_gyro_y`	TEXT NOT NULL,
            `segment_gyro_z`	TEXT NOT NULL,
            `segment_accl_x`	TEXT NOT NULL,
            `segment_accl_y`	TEXT NOT NULL,
            `segment_accl_z`	TEXT NOT NULL,
            `version`           REAL NOT NULL
        );
        """);
        self.conn.commit()

        # JSON encode segments and load them into the DB
        for segment in filtered_segments:
            b = {}

            for component in segment.filtered_data:
                b[component] = segment.filtered_data[component].tolist()
                b[component] = json.dumps(b[component])

            self.cursor.execute("INSERT INTO `" + table_name +
            """` (segment_gyro_x,segment_gyro_y,
                 segment_gyro_z,segment_accl_x,
                 segment_accl_y,segment_accl_z,
                 version)
                VALUES (?,?,?,?,?,?,?);
            """,[
            b["gyro_X"],
            b["gyro_Y"],
            b["gyro_Z"],
            b["accel_X"],
            b["accel_Y"],
            b["accel_Z"],
            SignalFilter.VERSION])
        self.conn.commit()

