#!/usr/bin/python
import commands, os
import numpy

#Model
amp = 4.2
omega = 4.9
force = 0.0
gam = 0.9
Dg = 0.001
Dp = 0
lmd = 0
comp = 0
fa = 0
fb = 0
mua = 0
mub = 0
mean = 0

#Simulation
dev = 1
block = 64
paths = 1024 #4096
periods = 10000 #2000
spp = 100 #200
algorithm = 'predcorr'
frac = 0.1
trans = int(frac*periods)
samples = spp

#Output
mode = 'moments'
points = 100
beginx = 0
endx = 0.2
domain = '1d'
domainx = 'f'
logx = 0
DIRNAME='./tests/jstatmech_predcorr/'
os.system('mkdir -p %s' % DIRNAME)

#os.system('rm -v %s*.dat %s*.png' % (DIRNAME, DIRNAME))

#fig 2a

force = 0
lmd = 0.1
domainx = 'p'
c = 0.2
beginx = 0
endx = c**2/lmd

#fig 2b & fig 3a

amp = 4.2
beginx = 0
for lmd in [0.1, 4, 16]:
    endx = c**2/lmd
    if lmd < 1:
        out = 'fig2b_lmd%s' % lmd
    else:
        out = 'fig3a_lmd%s' % lmd
    _cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)
os.system('gnuplot jstatmech_predcorr.plt')
os.system('mv -vf fig*.dat fig*.png %s' % DIRNAME)
