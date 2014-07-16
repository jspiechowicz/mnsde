set terminal pngcairo enhanced
set key outside center top horizontal samplen 2
set autoscale fix
set xzeroaxis
set ylabel '{/Symbol \341}v{/Symbol \361}' rotate by 360

set output 'fig2b.png'
plot [0:0.2] 'fig2b_lmd0.1.dat' using (sqrt($1*0.1)):2 title '{/Symbol l} = 0.1' with lines linetype 1 linewidth 3
unset output

set output 'fig3a.png'
plot [0:0.2] 'fig3a_lmd4.dat' using (sqrt($1*4)):2 title '{/Symbol l} = 4' with lines linetype 1 linewidth 3,\
'fig3a_lmd16.dat' using (sqrt($1*16)):2 title '{/Symbol l} = 16' with lines linetype 2 linewidth 3
unset output

exit gnuplot
