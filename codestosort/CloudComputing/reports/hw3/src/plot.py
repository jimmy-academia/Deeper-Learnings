import matplotlib.pyplot as plt

from thrift.transport import TSocket,TTransport
from thrift.protocol import TBinaryProtocol
from hbase import Hbase
from hbase.ttypes import ColumnDescriptor
from hbase.ttypes import Mutation
import csv
import os

import time
import logging
from tqdm import tqdm
import numpy as np
# table: station, column: attr, row: date

from datetime import datetime
from datetime import timedelta
oneday = timedelta(days=1)
firstday = datetime.strptime('2018/1/1' , '%Y/%m/%d')
lastday = datetime.strptime('2018/12/31' , '%Y/%m/%d')

ALL = []
NAME = []
Y = []
# datetime.strftime('%Y/%m/%d')
def plot_curve(X,y, name):
    # trans = mdates.strpdate2num('%Y/%m/%d')
    # day = [trans(i) for i in X]
    ALL.append(X)
    NAME.append(name)
    Y.append(y)
    plt.plot(X,y)
    plt.title(name+' PM2.5')
    plt.xlabel('day')
    plt.ylabel('value')
    plt.savefig(name+'.png')
    plt.close()

def plot_all():
    for X, y, name in zip(ALL, Y, NAME):
        plt.plot(X,y, label=name)
    plt.title('ALL regions PM2.5')
    plt.xlabel('day')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('all.png')
    plt.close()

def main():
    socket = TSocket.TSocket('127.0.0.1',9090)
    socket.setTimeout(5000)
    transport = TTransport.TBufferedTransport(socket)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = Hbase.Client(protocol)
    socket.open()

    table_list = client.getTableNames()
    start = time.time()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('Initiating task: Plot Taiwan Air Quality!')

    Attributes = ['AMB_TEMP','CO','NO','NO2','NOx','O3','PM10','PM2.5','RAINFALL','RAIN_COND','UVB',
        'RH','SO2','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR','CH4','NMHC','THC','PH_RAIN']

    ## row key is dates '2018/1/1' '2018/12/31'
    count = 0
    for table in table_list:
        logging.info('processing %s'%table)

        X = []
        y = []
        iterday = firstday
        i = 1
        attr = ['PM2.5:%d'%i for i in range(1,23)]
        while iterday <= lastday:
            row = iterday.strftime('%Y/%m/%d')
            DATA = client.getRow(table, row)
            columns = DATA[0].columns
            vals =[]
            for at in attr:
                try:
                    val = columns[at].value
                    vals.append(float(val))
                except:
                    pass

            mean = np.mean(vals)
            X.append(i)
            y.append(mean)
            i+=1
            iterday = iterday+oneday

        plot_curve(X,y,table)
    plot_all()

if __name__ == '__main__':
    main()