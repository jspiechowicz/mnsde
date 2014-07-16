set terminal pngcairo enhanced
set key outside center top horizontal samplen 2
set xzeroaxis
set ylabel '{/Symbol \341}v{/Symbol \361}' rotate by 360

set output 'prl.png'
set xlabel 'f'
plot 'prl.dat' using 1:2 notitle with lines linetype 1 linewidth 3
unset output

exit gnuplot
