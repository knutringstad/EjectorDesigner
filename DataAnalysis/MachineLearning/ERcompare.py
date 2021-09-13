import matplotlib.pyplot as plt
import numpy as np

Case = [1,2,3]
ERCFD= [0.598,0.753,0.793]
ERGPR= [0.601,0.714,0.742]


N = 3
a = [ERCFD,ERGPR]



ind = np.arange(N)    # the x locations for the groups
width = 0.45      # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()


p1 = ax.bar(ind, ERCFD, width, label='cfd')
p2 = ax.bar(ind+width, ERGPR, width, label='gpr')


ax.set_ylabel('Entrainment ratio')
ax.set_ylim([0,0.9])
ax.set_xticks(ind+width/2)
ax.set_xticklabels( ('Geometry 1', 'Geometry 2', 'Geometry 3') )

ax.legend( (p1[0], p2[0]), ('CFD', 'GPR'), loc="upper left")

for i in range(3):
    ax.text(ind[i]-width/3, ERCFD[i] + 0.03, str(ERCFD[i]), color='k', fontweight='bold')
    ax.text(ind[i]+width-width/3, ERGPR[i] + 0.03, str(ERGPR[i]), color='k', fontweight='bold')

plt.show()