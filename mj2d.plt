set terminal pngcairo enhanced

set autoscale fix
set pm3d interpolate 0,0 corners2color mean map 
set samples 100
set isosamples 100

set output 'fig5.1a.png'
set xlabel '{/Symbol g}'
set ylabel 'a' rotate by 360
stats 'fig5.1a.dat' using 3 nooutput
set palette color defined (STATS_min "blue", 0 "white", STATS_max "red")
splot 'fig5.1a.dat' using 1:2:3 notitle
unset output

set output 'fig5.1b.png'
set xlabel '{/Symbol w}'
set ylabel 'a' rotate by 360
stats 'fig5.1b.dat' using 3 nooutput
set palette color defined (STATS_min "blue", 0 "white", STATS_max "red")
splot 'fig5.1b.dat' using 1:2:3 notitle
unset output

set output 'fig5.2b.png'
set xlabel 'a'
set ylabel '{/Symbol g}' rotate by 360
stats 'fig5.2b.dat' using 3 nooutput
set palette color defined (STATS_min "blue", 0 "white", STATS_max "red")
splot 'fig5.2b.dat' using 1:2:3 notitle
unset output

set output 'fig5.2c.png'
set xlabel 'a'
set ylabel '{/Symbol g}' rotate by 360
stats 'fig5.2c.dat' using 3 nooutput
set palette color defined (STATS_min "blue", 0 "white", STATS_max "red")
splot 'fig5.2c.dat' using 1:2:3 notitle
unset output

set output 'fig5.2d.png'
set xlabel 'a'
set ylabel '{/Symbol g}' rotate by 360
stats 'fig5.2d.dat' using 3 nooutput
set palette color defined (STATS_min "blue", 0 "white", STATS_max "red")
splot 'fig5.2d.dat' using 1:2:3 notitle
unset output

set output 'fig5.6b.png'
set xlabel 'a'
set ylabel '{/Symbol g}' rotate by 360
stats 'fig5.6_Dg0.0001.dat' using 3 nooutput
set palette color defined (STATS_min "blue", 0 "white", STATS_max "red")
splot 'fig5.6_Dg0.0001.dat' using 1:2:3 notitle
unset output

set output 'fig5.6c.png'
set xlabel 'a'
set ylabel '{/Symbol g}' rotate by 360
stats 'fig5.6_Dg0.0005.dat' using 3 nooutput
set palette color defined (STATS_min "blue", 0 "white", STATS_max "red")
splot 'fig5.6_Dg0.0005.dat' using 1:2:3 notitle
unset output

set output 'fig5.6d.png'
set xlabel 'a'
set ylabel '{/Symbol g}' rotate by 360
stats 'fig5.6_Dg0.001.dat' using 3 nooutput
set palette color defined (STATS_min "blue", 0 "white", STATS_max "red")
splot 'fig5.6_Dg0.001.dat' using 1:2:3 notitle
unset output

exit gnuplot
