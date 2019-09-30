import sys
from datetime import datetime
for line in sys.stdin:
        line = line.strip()
        if line == '':
                continue
        begin = line.find('[')
        end = line.find(']')
        record = line[begin+1:end-6]
        #print(record)
        record = str(datetime.strptime(record,"%d/%b/%Y:%H:%M:%S"))
        record = record[:len(record)-6]+':00:00'
        #sys.stdout.write(record+'\t1')
        print(record+'\t1')

