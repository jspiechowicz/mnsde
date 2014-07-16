#!/usr/bin/python
import commands, os
import numpy

pipi = 2.0*numpy.pi

#Model
amp = 0
omega = 0.6*pipi
force = 0.05*pipi
gam = 0
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
dev = 2
block = 64
paths = 1024
periods = 1000
spp = 100
algorithm= 'predcorr'
trans = 0.1
samples = 100

#Output
mode = 'moments'
points = 100
beginx = 0.1*pipi
endx = 1.0*pipi
beginy = 0.0
endy = 6.0*pipi
domain = '2d'
domainx = 'g'
domainy = 'a'
logx = 0
logy = 0
DIRNAME='./tests/mj2d/'
os.system('mkdir -p %s' % DIRNAME)

os.system('rm -v %s*.dat %s*.png' % (DIRNAME, DIRNAME))

#fig 5.1a

out = 'fig5.1a'
_cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s --domainy=%s --logy=%d --beginy=%s --endy=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, domainy, logy, beginy, endy, out)
output = open('%s.dat' % out, 'w')
print >>output, '#%s' % _cmd
output.close()
print _cmd
cmd = commands.getoutput(_cmd)

#fig 5.1b

gam = 0.19*pipi
beginx = 0.1*pipi
endx = 2.0*pipi
domainx = 'w'

out = 'fig5.1b'
_cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s --domainy=%s --logy=%d --beginy=%s --endy=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, domainy, logy, beginy, endy, out)
output = open('%s.dat' % out, 'w')
print >>output, '#%s' % _cmd
output.close()
print _cmd
cmd = commands.getoutput(_cmd)

#fig 5.2b

omega = 0.6*pipi
force = 0.001*pipi
beginx = 1.4*pipi
endx = 2.0*pipi
beginy = 0.05*pipi
endy = 0.25*pipi
domainx = 'a'
domainy = 'g'

out = 'fig5.2b'
_cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s --domainy=%s --logy=%d --beginy=%s --endy=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, domainy, logy, beginy, endy, out)
output = open('%s.dat' % out, 'w')
print >>output, '#%s' % _cmd
output.close()
print _cmd
cmd = commands.getoutput(_cmd)

#fig 5.2c

force = 0.01*pipi

out = 'fig5.2c'
_cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s --domainy=%s --logy=%d --beginy=%s --endy=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, domainy, logy, beginy, endy, out)
output = open('%s.dat' % out, 'w')
print >>output, '#%s' % _cmd
output.close()
print _cmd
cmd = commands.getoutput(_cmd)

#fig 5.2d

force = 0.06*pipi

out = 'fig5.2d'
_cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s --domainy=%s --logy=%d --beginy=%s --endy=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, domainy, logy, beginy, endy, out)
output = open('%s.dat' % out, 'w')
print >>output, '#%s' % _cmd
output.close()
print _cmd
cmd = commands.getoutput(_cmd)

#fig 5.6b-d

force = 0.01*pipi

for Dg in [0.0001, 0.0005, 0.001]:
    out = 'fig5.6_Dg%s' % Dg
    _cmd = './underdamped --dev=%d --amp=%s --omega=%s --force=%s --gam=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --fa=%s --fb=%s --mua=%s --mub=%s --mean=%s --block=%d --paths=%d --periods=%s --trans=%s --spp=%d --samples=%d --algorithm=%s --mode=%s --domain=%s --domainx=%s --logx=%d --points=%d --beginx=%s --endx=%s --domainy=%s --logy=%d --beginy=%s --endy=%s >> %s.dat' % (dev, amp, omega, force, gam, Dg, Dp, lmd, comp, fa, fb, mua, mub, mean, block, paths, periods, trans, spp, samples, algorithm, mode, domain, domainx, logx, points, beginx, endx, domainy, logy, beginy, endy, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)

os.system('gnuplot mj2d.plt')
os.system('mv -v *.dat *.png %s' % DIRNAME)
