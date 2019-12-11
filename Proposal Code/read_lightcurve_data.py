import csv
from datetime import datetime
from numpy import *
import matplotlib.pyplot as plt

class SpacecraftObservation():

    def __init__(self, csv_file):

        self.Passes = self.parse_passes(csv_file)

    def parse_passes(self, csv_file):

        pass_list = []

        with open(csv_file) as file:
            
            rows = csv.reader(file, delimiter = ',')
            for num, row in enumerate(rows):
                if num >= 13:
                    Pass = {}
                    Pass['REQUEST_ID'] = row[0]
                    Pass['NORAD_ID'] = row[1]
                    Pass['EPOCH'] = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S.%f')
                    Pass['TIME'] = asarray([float(x)/1000 for x in row[3::6] if x != ''])
                    Pass['AZ'] = asarray([float(x) for x in row[4::6]])
                    Pass['EL'] = asarray([float(x) for x in row[5::6]])
                    Pass['RA'] = asarray([float(x) for x in row[6::6]])
                    Pass['DEC'] = asarray([float(x) for x in row[7::6]])
                    Pass['MAG'] = asarray([float(x) if x != '' else 0 for x in row[8::6]])
                    Pass['TLE'] = None
                    pass_list.append(Pass)

        return pass_list


    def find_tle(self, tle_file):

        file = open(tle_file,'r')
        lines = file.readlines()
        for __pass in self.Passes:

            date = __pass['EPOCH'].timetuple()
            datenum = (date.tm_year%100)*1000
            datenum += date.tm_yday
            datenum += (date.tm_hour + date.tm_min/60 + date.tm_sec/3600)/24
            
            for indx in range(0, len(lines), 2):
                tle_datenum = float(lines[indx].split()[3])
                if tle_datenum > datenum:
                    __pass['TLE'] = [lines[indx-2].replace('\n',''),
                                     lines[indx-1].replace('\n','')]
                    break



















if __name__ == '__main__':
    test = SpacecraftObservation('CalPolySLO_obs.csv')
    test.find_tle('sat41788.txt')
    p = test.Passes[0]
    print(p['TIME'])
    plt.figure()
    plt.plot(p['TIME'], p['MAG'],'.')
    plt.title('CANX-7 Light Curve '+str(p['EPOCH']))
    plt.xlabel('Time [s]')
    plt.ylabel('Photometric Magnitude')
    plt.show()
    # for p in test.Passes:
    #     print(p['TLE'])
    # for i, p in enumerate(test.Passes):
    #     plt.figure()
    #     plt.plot(p['TIME'][:len(p['MAG'])], p['MAG'])
    #     plt.title(str(i))
    # plt.show()