set terminal pngcairo enhanced
set key outside center top horizontal samplen 2
set autoscale fix
set xzeroaxis
set ylabel '{/Symbol \341}v{/Symbol \361}' rotate by 360

set output 'fig5.8.png'
set xlabel 'f'
plot 'fig5.8_Dg0.dat' using 1:2 title 'D_G = 0' with lines linetype 1 linewidth 3,\
'fig5.8_Dg0.0001.dat' using 1:2 title 'D_G = 10^{-4}' with lines linetype 2 linewidth 3
unset output

set output 'fig5.10.png'
set xlabel 'f'
plot 'fig5.10_Dg0.dat' using 1:2 title 'D_G = 0' with lines linetype 1 linewidth 3,\
'fig5.10_Dg0.0007.dat' using 1:2 title 'D_G = 7*10^{-4}' with lines linetype 2 linewidth 3
unset output

set output 'fig5.11.png'
set xlabel 'f'
plot 'fig5.11_Dg0.dat' using 1:2 title 'D_G = 0' with lines linetype 1 linewidth 3,\
'fig5.11_Dg0.0003.dat' using 1:2 title 'D_G = 3*10^{-4}' with lines linetype 2 linewidth 3
unset output

set output 'fig5.13.png'
set xlabel 'f'
plot [0:1.25] 'fig5.13_Dg0.dat' using 1:2 title 'D_G = 0' with lines linetype 1 linewidth 3,\
'fig5.13_Dg0.0001.dat' using 1:2 title 'D_G = 10^{-4}' with lines linetype 2 linewidth 3
unset output

exit gnuplot
