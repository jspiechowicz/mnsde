#!/usr/bin/python
import commands, os
import numpy

pipi = 2.0*numpy.pi

#Model
amp = 1.62151*pipi
omega = 0.6*pipi
force = 0
gam = 0.13*pipi
Dg = 0
Dp = 0
lmd = 0
comp = 0
fa = 0
fb = 0
mua = 0
mub = 0
mean = 0

#Simulation
dev = 0
block = 64
paths = 4096
periods = 2000
spp = 200
algorithm = 'predcorr'
trans = 0.1
samples = 200

#Output
mode = 'moments'
points = 100
beginx = 0
endx = 0.25
domain = '1d'
domainx = 'f'
logx = 0
DIRNAME='./tests/mj/'
os.system('mkdir -p %s' % DIRNAME)

os.system('rm -v %s*.dat %s*.png' % (DIRNAME, DIRNAME))

#fig 5.8

for Dg in [0, 0.0001]:
    out = 'fig5.8_Dg%s' % Dg
    _cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)

#fig 5.10

amp = 1.721042*pipi
gam = 0.165832*pipi
endx = 0.5

for Dg in [0, 0.0007]:
    out = 'fig5.10_Dg%s' % (Dg)
    _cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)

#fig 5.11

amp = 1.997595*pipi
gam = 0.098096*pipi

for Dg in [0, 0.0003]:
    out = 'fig5.11_Dg%s' % (Dg)
    _cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)

#fig 5.13

amp = 0.668*pipi
omega = 0.78*pipi
gam = 0.143*pipi
endx = 1.25

for Dg in [0, 0.0001]:
    out = 'fig5.13_Dg%s' % (Dg)
    _cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)

os.system('gnuplot mj.plt')
os.system('mv -vf fig*.dat fig*.png %s' % DIRNAME)
