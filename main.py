from singleh import *
nn = NN();
# print('\n'.join([''.join([' {:4}'.format(item) for item in row])
#      for row in nn.ws_o]))
digs = ldigits()
for x in xrange(200):
      dig = int(random.random() * 10)
      nn.epoch(digs[dig],dig-1)
print [round(f,4) for f in nn.epoch(digs[4],3)]

print
#print nn.error
