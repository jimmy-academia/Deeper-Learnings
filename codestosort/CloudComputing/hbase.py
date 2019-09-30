
from thrift.transport import TSocket,TTransport
from thrift.protocol import TBinaryProtocol
from hbase import Hbase

# socket = TSocket.TSocket('192.168.0.156',9090)
# socket.setTimeout(5000)
# transport = TTransport.TBufferedTransport(socket)
# protocol = TBinaryProtocol.TBinaryProtocol(transport)
# client = Hbase.Client(protocol)
# socket.open()

socket = TSocket.TSocket('10.0.0.5',9090)
socket.setTimeout(5000)
transport = TTransport.TBufferedTransport(socket)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = Hbase.Client(protocol)
socket.open()

socket = TSocket.TSocket('172.18.0.3',9090)
socket.setTimeout(5000)
transport = TTransport.TBufferedTransport(socket)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = Hbase.Client(protocol)
socket.open()

socket = TSocket.TSocket('127.0.0.1',9090)
socket.setTimeout(5000)
transport = TTransport.TBufferedTransport(socket)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = Hbase.Client(protocol)
socket.open()


print(client.getTableNames())
print(client.get('test','row1','cf:a'))



