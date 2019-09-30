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

# table: station, column: attr, row: date

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

    logging.info('Initiating task: Taiwan Air Quality!')

    Attributes = ['AMB_TEMP','CO','NO','NO2','NOx','O3','PM10','PM2.5','RAINFALL','RAIN_COND','UVB',
        'RH','SO2','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR','CH4','NMHC','THC','PH_RAIN']

    csvfiles = [filename for filename in os.listdir(os.getcwd()) if filename.endswith('.csv')]
    logging.info(str(csvfiles))

    InsertCounts = 0

    for file in csvfiles:
        with open(file, newline='') as f:
            frames = csv.reader(f)
            table_Name = ''
            logging.info("Start reading {0}".format(file))

            Column_Descriptors = []
            ctr = 0

            # length = sum(1 for row in frames)
# 
            # for frame in tqdm(frames, total=length):
            for frame in tqdm(frames):
                if ctr == 0:
                    ctr += 1
                    continue
                elif ctr == 1:
                    ctr += 1
                    table_Name = str(str.encode(frame[1],'utf-8')).replace('\\',"")
                    table_Name = table_Name.replace("b","")
                    table_Name = table_Name.replace("'","")
                    if table_Name not in table_list:
                        for type in Attributes:
                            Column_Descriptors.append(ColumnDescriptor(name=type))
                        client.createTable(table_Name,Column_Descriptors)
                        logging.info('Build Table : {0}'.format(table_Name))
                    else:
                        logging.info('Table {0} already exist, no need to create'.format(table_Name))

                # ['2018/01/02', 'iilan', 'NOx', '5.1', '4.4', '3.5', '2.1', '2.5', '3.2', '4.6', '15', 
                # '13', '11', '7', '6.8', '7.1', '13', '13', '12', '13', '16', '24', '23', '20', '24', '18', '13']

                for i in range(3,26):
                    qualifier = i-2
                    value = frame[i]    
                    row = frame[0]      # date
                    column = frame[2]   # attr
                    mutate = Mutation(column=column+':'+str(qualifier),value=value)
                    client.mutateRow(table_Name, frame[0], [mutate])
                    InsertCounts += 1


    end = time.time()

    logging.info("================Insert Done================\n")
    logging.info("totalInsertCount: {0}, totalTimeSpend: {1}\n".format(InsertCounts,end-start))
    logging.info(client.getTableNames())



if __name__ == '__main__':
    main()