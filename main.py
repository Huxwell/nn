from singleh import *
nn = NN();
# print('\n'.join([''.join([' {:4}'.format(item) for item in row])
#      for row in nn.ws_o]))
digs = ldigits()
for x in xrange(100):
      dig = int(random.random() * 10)
      nn.epoch(digs[dig],dig)
print [round(f,1000) for f in nn.epoch(digs[0],0)]