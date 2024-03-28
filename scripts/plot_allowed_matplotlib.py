import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.style as style
style.use('seaborn-colorblind')

# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])
z = np.array([10, 20, 30, 40, 50])  # Z values to represent colors

points_files = ['oks_fullscan_ht_new.dat', 'oks_minus_ht_new.dat', 'oks_smallsa_ht_new.dat', 'oks_smallsam_ht_new.dat']

data_1 = {}
data_2 = {}

data_failed_1 = {}

wmass_x = []
wmass_y = []
wmass_y_lower = []
w_mass_file = 'res.dat_new'
with open(w_mass_file,'r') as file:

    for line in file:
        split_line=line.split()
        wmass_x.append(float(split_line[0]))
        wmass_y.append(float(split_line[1]))
        wmass_y_lower.append(-float(split_line[1]))

for points_file in points_files:
    with open(points_file,'r') as file:
    
        for line in file:
    
            split_line = line.split()
            mH = float(split_line[1])
            sina = float(split_line[2])
            tanb = float(split_line[3])
    

            if not split_line[-2]=='1':

                if sina>0 :
                    if mH in data_failed_1:
                        data_failed_1[mH] = min(data_failed_1[mH], sina)
                    else:
                        data_failed_1[mH] = sina
            
            passed = split_line[-1]=='1' and split_line[-2]=='1'

            if not passed:
                continue

            if (mH,sina) not in data_1 or tanb > data_1[(mH,sina)]:
                data_1[(mH,sina)] = tanb
            if (mH,tanb) not in data_2 or abs(sina) > abs(data_2[(mH,tanb)]):
                data_2[(mH,tanb)] = abs(sina)

#data_failed_1_x = []
#data_failed_1_y = []
#for key in data_failed_1:
#  mH = key
#  sina = data_failed_1[key]
#  data_failed_1_x.append(mH)
#  data_failed_1_y.append(sina)


x1 = []
y1 = []
z1 = []        

for key in data_1:
  mH = key[0]
  sina = key[1]
  max_tanb = data_1[key]
  x1.append(mH)
  y1.append(sina)
  z1.append(max_tanb)

x2=[]
y2=[]
z2=[]
for key in data_2:
  mH = key[0]
  tanb = key[1]
  max_sina = data_2[key]
  x2.append(mH)
  y2.append(tanb)
  z2.append(max_sina)

#cmap = 'Blues'
cmap = 'viridis_r'
#cmap = 'cividis_r'
#cmap = 'Purples'

# plot sina vs mH
plt.scatter(x1, y1, c=z1, cmap=cmap, marker='o',s=10)
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction='in',labelsize=12,length=8)
ax.set_xlim([min(x1), max(x1)])
ax.set_ylim([-0.3, 0.3])
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', which='minor', length=4, width=1)
ax.fill_between(wmass_x, wmass_y, 5, color='red', alpha=0.3,label=r'$m_{\mathrm{W}}$ constraint')
#ax.fill_between(data_failed_1_x, data_failed_1_y, 5, color='blue', alpha=0.3,label=r'test')
ax.fill_between(wmass_x, wmass_y_lower, -5, color='red', alpha=0.3)
ax.legend(loc='upper left', fontsize=14, bbox_to_anchor=(0., 1.15), ncol=2, frameon=False)
cb = plt.colorbar()
cb.set_label(r'Max. $\tan\beta$')
cb.ax.yaxis.label.set_fontsize(14)
cb.ax.yaxis.set_tick_params(labelsize=12)
#cb.set_clim(min(z1), max(z1))
plt.clim(min(z1),max(z1))
plt.ylabel(r'$\sin\alpha$', fontsize=14)
plt.xlabel(r'$m_{\mathrm{H}}$ (GeV)', fontsize=14)
plt.savefig('allowed_points_sina_vs_mH_maxtanb.pdf')
plt.close()

# plot tanb vs mH
plt.scatter(x2, y2, c=z2, cmap=cmap, marker='o',s=10)
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction='in',labelsize=12, length=8)
ax.set_xlim([min(x1), max(x1)])
ax.set_ylim([0., 4.5])
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', which='minor', length=4, width=1)
cb = plt.colorbar()
cb.set_label(r'Max. $|\sin\alpha|$')
cb.ax.yaxis.label.set_fontsize(14)
cb.ax.yaxis.set_tick_params(labelsize=12)
plt.clim(min(z2),max(z2))
plt.ylabel(r'$\tan\beta$', fontsize=14)
plt.xlabel(r'$m_{\mathrm{H}}$ (GeV)', fontsize=14)
plt.savefig('allowed_points_tanb_vs_mH_maxsina.pdf')

