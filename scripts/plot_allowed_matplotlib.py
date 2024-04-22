import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.style as style
import os
style.use('seaborn-colorblind')

points_files = ['oks_fullscan_ht_new.dat', 'oks_minus_ht_new.dat', 'oks_smallsa_ht_new.dat', 'oks_smallsam_ht_new.dat']

#210610361 = https://arxiv.org/abs/2106.10361, CMS H->hh->bbtautau
#1804019391 = https://arxiv.org/abs/1804.01939, CMS H->ZZ->4l,2l2q,2l2n (2016-only)
#2009147911,2009147912 = https://arxiv.org/abs/2009.14791v1 (v1), ATLAS H->ZZ->4l/2l2nu 
#2004146361 = https://arxiv.org/abs/2004.14636, ATLAS H->WW,ZZ -> semileptonic - it is a combination  of both!
#1808023801 = https://arxiv.org/abs/1808.02380: ATLAS H->VV,hh - figure 5a which is H->VV combination
exp_exclusions = ['2009147912', '210610361', '1804019391', '2004146361','1808023801','2009147911', 'plusexcl.out']

#H->ZZ = 2009147912, 2009147911, 1804019391
#H->hh = 210610361
#H->WW/ZZ->2l2nu = 2004146361
#H->VV,hh = 1808023801

# H->ZZ = 2009147911,2009147912,1804019391
#H->VV = 2004146361, 1808023801
  

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

all_points = {}

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

            if mH not in all_points:
                all_points[mH] = []
            all_points[mH].append((sina,tanb))

            if (mH,sina) not in data_1 or tanb > data_1[(mH,sina)]:
                data_1[(mH,sina)] = tanb
            if (mH,tanb) not in data_2 or abs(sina) > abs(data_2[(mH,tanb)]):
                data_2[(mH,tanb)] = abs(sina)

exclusions_points_map_sina = {}


points = np.array(all_points[280.])
tanb=1

#x=[]
#y=[]
#for p in points:
#  x.append(p[0])
#  y.append(p[1])
#plt.scatter(x, y, color='blue',marker='o',s=10)
#plt.title('Allowed points for mH=180 in oks.tar.gz')
#plt.ylabel(r'$\tan\beta$', fontsize=14)
#plt.xlabel(r'$\sin\alpha$', fontsize=14)
#plt.savefig('allowed_points_for_mH280.pdf')
#plt.close()

all_points_map = {}

for ex in exp_exclusions:

    for sign in ['plus','minus']:
        key = ex+'_'+sign
        if not os.path.isfile('%(sign)s/%(ex)s' % vars()): continue
        exclusions_points_map_sina[key] = {}
        all_points_x = []
        all_points_y = []
        with open('%(sign)s/%(ex)s' % vars() ,'r') as file:
            for line in file:
                split_line = line.split()
                mH = float(split_line[1])
                sina = float(split_line[2])
                all_points_x.append(mH) 
                all_points_y.append(sina) 

                # as points are spaced by 0.01 in sina, we +/- 0.005 (half the width between points)
                offset_up = 0.005
                offset_down = 0.005
                if mH>260 and mH<330 and (key == "2009147912_minus" or (mH<330 and key == "1808023801_minus")): offset_down = 0.01 # increasing this for cosmetic issues as you can see artificial white space between these bounds and W+mass due to finite point spacing
                if mH not in exclusions_points_map_sina[key]:
                    exclusions_points_map_sina[key][mH] = (sina-offset_down,sina+offset_up)
                
                exclusions_points_map_sina[key][mH] = (min(exclusions_points_map_sina[key][mH][0],sina-offset_down), max(exclusions_points_map_sina[key][mH][1],sina+offset_up))

        #plt.scatter(all_points_x, all_points_y, color='blue', marker='o',s=6)
        #ax = plt.gca()
        #plt.xlabel(r'$m_{H}$', fontsize=14)
        #plt.ylabel(r'$\sin(\alpha)$ (GeV)', fontsize=14)
        #plt.savefig('excluded_points_%(key)s.pdf' % vars())
        #plt.close()


for key in exclusions_points_map_sina:
    x_list = list(exclusions_points_map_sina[key].keys())
    y_tuple_list = list(exclusions_points_map_sina[key].values())
    if len(x_list) == 0 or len(y_tuple_list) == 0:
        exclusions_points_map_sina[key] = [[],[],[]]
        continue
    y_max_list = []
    y_min_list = []
    for yt in y_tuple_list:
        y_min_list.append(yt[0])  
        y_max_list.append(yt[1])  
    sorted_lists = sorted(zip(x_list, y_min_list, y_max_list))
    x_sorted, y_min_sorted, y_max_sorted  = zip(*sorted_lists)

    x_sorted = list(x_sorted)
    y_min_sorted = list(y_min_sorted)
    y_max_sorted = list(y_max_sorted)
    exclusions_points_map_sina[key] = [x_sorted,y_min_sorted, y_max_sorted]


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

def CombineExclusions(cmb_list=[], out_name=''):
    cmb_map = {}
    x_list = []
    y_min_list = []
    y_max_list = []
    for ex in cmb_list:
        for i, mH in enumerate(exclusions_points_map_sina[ex][0]):
            y_min = exclusions_points_map_sina[ex][1][i]
            y_max = exclusions_points_map_sina[ex][2][i]
            if mH not in cmb_map:
                cmb_map[mH] = (y_min,y_max)
            else: 
                cmb_map[mH] = (min(y_min, cmb_map[mH][0]), max(y_max, cmb_map[mH][1]))

    x_list = list(cmb_map.keys())
    y_tuple_list = list(cmb_map.values())
    y_max_list = []
    y_min_list = []
    for yt in y_tuple_list:
        y_min_list.append(yt[0])
        y_max_list.append(yt[1])
    sorted_lists = sorted(zip(x_list, y_min_list, y_max_list))
    x_sorted, y_min_sorted, y_max_sorted  = zip(*sorted_lists)

    x_sorted = list(x_sorted)
    y_min_sorted = list(y_min_sorted)
    y_max_sorted = list(y_max_sorted)
    exclusions_points_map_sina[out_name] = [x_sorted,y_min_sorted, y_max_sorted]

# plot sina vs mH
plt.scatter(x1, y1, c=z1, cmap=cmap, marker='o',s=6)


ax = plt.gca()
ax.tick_params(axis='both', which='both', direction='in',labelsize=12,length=8)
ax.set_xlim([min(x1), max(x1)])
ax.set_ylim([-0.3, 0.3])
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', which='minor', length=4, width=1)


CombineExclusions(['2009147912_plus', '2009147911_plus', '1804019391_plus'], 'HZZ_plus')
CombineExclusions(['2009147912_minus', '2009147911_minus', '1804019391_minus'], 'HZZ_minus')
CombineExclusions(['2004146361_plus'], 'HVV_plus')
CombineExclusions(['2004146361_minus','1808023801_minus'], 'HVV_minus')

# exp constraints
ax.fill_between(exclusions_points_map_sina['HZZ_plus'][0], exclusions_points_map_sina['HZZ_plus'][1], exclusions_points_map_sina['HZZ_plus'][2], color='red', alpha=0.3,label=r'H$\rightarrow$ZZ')
ax.fill_between(exclusions_points_map_sina['HZZ_minus'][0], exclusions_points_map_sina['HZZ_minus'][1], exclusions_points_map_sina['HZZ_minus'][2], color='red', alpha=0.3)
ax.fill_between(exclusions_points_map_sina['HVV_plus'][0], exclusions_points_map_sina['HVV_plus'][1], exclusions_points_map_sina['HVV_plus'][2], color='blue', alpha=0.3,label=r'H$\rightarrow$VV')
ax.fill_between(exclusions_points_map_sina['HVV_minus'][0], exclusions_points_map_sina['HVV_minus'][1], exclusions_points_map_sina['HVV_minus'][2], color='blue', alpha=0.3)
ax.fill_between(exclusions_points_map_sina['210610361_plus'][0], exclusions_points_map_sina['210610361_plus'][1], exclusions_points_map_sina['210610361_plus'][2], color='yellow', alpha=0.3, label=r'H$\rightarrow$hh')


ax.fill_between(wmass_x, wmass_y, 5, color='darkviolet', alpha=1.0,label=r'$m_{\mathrm{W}}$')
ax.fill_between(wmass_x, wmass_y_lower, -5, color='darkviolet', alpha=1.0)

#redraw points so they appear on top
plt.scatter(x1, y1, c=z1, cmap=cmap, marker='o',s=6)

ax.legend(loc='upper left', fontsize=14, bbox_to_anchor=(0., 1.19), ncol=2, frameon=False)
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

