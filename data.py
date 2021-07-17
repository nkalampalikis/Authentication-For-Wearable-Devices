import numpy as np


# Eventually will be a loader for a DB
class Data:
    def get_hz(self):
        return self.hz

    def __init__( self, data_a, data_g, hz):
        self.hz = hz

        assert len( data_g ) == len( data_a )
        assert data_g[ "TS" ][ 0 ] == data_a[ "TS" ][ 0 ]

        array = []
        for i in range(0, len(data_g["TS"])):
            TS = int( data_g["TS"][i] )
            accel_X = float( data_a["X"][i] )
            accel_Y = float( data_a["Y"][i] )
            accel_Z = float( data_a["Z"][i] )
            gyro_X = float( data_g["X"][i] )
            gyro_Y = float( data_g["Y"][i] )
            gyro_Z = float( data_g["Z"][i] )
            array.append( ( TS, accel_X, accel_Y, accel_Z,
                                gyro_X,  gyro_Y,  gyro_Z ) )

        self.dtype = [
                ('TS', 'float64'),
                ('accel_X', 'float64'),
                ('accel_Y', 'float64'),
                ('accel_Z', 'float64'),
                ('gyro_X', 'float64'),
                ('gyro_Y', 'float64'),
                ('gyro_Z', 'float64')
                ]
        data = np.array( array, self.dtype )

        # # Set the start time to 0
        # for i in range(0, len(data)):
        #     data["TS"] -= data["TS"][0]

        self.data = data

    def get_data( self ):
        return self.data

    # Return number of seconds in this sample
    def seconds( self ):
        size = len(self.data)
        seconds = size / self.get_hz()
        return seconds