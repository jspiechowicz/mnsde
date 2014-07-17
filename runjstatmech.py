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
spp = 800
algorithm = 'euler'
frac = 0.1
trans = int(frac*periods)
samples = spp/2

#Output
mode = 'moments'
points = 100
beginx = 0
endx = 0.2
domain = '1d'
domainx = 'f'
logx = 0
DIRNAME='./tests/jstatmech/'
os.system('mkdir -p %s' % DIRNAME)

#os.system('rm -v %s*.dat %s*.png' % (DIRNAME, DIRNAME))

#fig 2a

force = 0
lmd = 0.1
domainx = 'p'
c = 0.2
beginx = 0
endx = c**2/lmd

for amp in [3.1, 4.2, 4.4]:  
    out = 'fig2a_a%s' % amp
    _cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)

#fig 2b & fig 3a

amp = 4.2
beginx = 0

for lmd in [0.1, 0.2, 0.3, 4, 16, 64, 512]:
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

#fig 3b

domainx = 'D'
beginx = -6
endx = 0
logx = 1
for lmd in [4, 512]:
    if lmd == 4:
        Dp = 6.0e-4
    elif lmd == 512:
        Dp = 2.0e-5
    out = 'fig3b_lmd%s' % lmd
    _cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)

#fig 4a

Dg = 0.001
domainx = 'p'
beginx = -6
endx = -1
for lmd in [0.1, 1, 10, 100]:
    out = 'fig4a_lmd%s' % lmd
    _cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)

#fig 4b

domainx = 'l'
beginx = -1
endx = 3
for Dp in [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3]:
    out = 'fig4b_Dp%s' % Dp
    _cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)

os.system('gnuplot jstatmech.plt')
os.system('mv -vf fig*.dat fig*.png %s' % DIRNAME)
