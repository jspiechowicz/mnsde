set terminal pngcairo enhanced
set key outside center top horizontal samplen 2
set autoscale fix
set xzeroaxis
set ylabel '{/Symbol \341}v{/Symbol \361}' rotate by 360

set output 'fig2a.png'
set xlabel '{/Symbol \341}{/Symbol h}{/Symbol \361}'
plot [0:0.2] 'fig2a_a3.1.dat' using (sqrt($1*0.1)):2 title 'a = 3.1' with lines linetype 1 linewidth 3,\
'fig2a_a4.2.dat' using (sqrt($1*0.1)):2 title 'a = 4.2' with lines linetype 2 linewidth 3,\
'fig2a_a4.4.dat' using (sqrt($1*0.1)):2 title 'a = 4.4' with lines linetype 3 linewidth 3
unset output

set output 'fig2b.png'
plot [0:0.2] 'fig2b_lmd0.1.dat' using (sqrt($1*0.1)):2 title '{/Symbol l} = 0.1' with lines linetype 1 linewidth 3,\
'fig2b_lmd0.2.dat' using (sqrt($1*0.2)):2 title '{/Symbol l} = 0.2' with lines linetype 2 linewidth 3,\
'fig2b_lmd0.3.dat' using (sqrt($1*0.3)):2 title '{/Symbol l} = 0.3' with lines linetype 3 linewidth 3
unset output

set output 'fig3a.png'
plot [0:0.2] 'fig3a_lmd4.dat' using (sqrt($1*4)):2 title '{/Symbol l} = 4' with lines linetype 1 linewidth 3,\
'fig3a_lmd16.dat' using (sqrt($1*16)):2 title '{/Symbol l} = 16' with lines linetype 2 linewidth 3,\
'fig3a_lmd64.dat' using (sqrt($1*64)):2 title '{/Symbol l} = 64' with lines linetype 3 linewidth 3,\
'fig3a_lmd512.dat' using (sqrt($1*512)):2 title '{/Symbol l} = 512' with lines linetype 4 linewidth 3
unset output

set logscale x
set format x '10^{%L}'
set output 'fig3b.png'
set xlabel 'D_G'
plot 'fig3b_lmd4.dat' using 1:2 title '{/Symbol l} = 4' with lines linetype 1 linewidth 3,\
'fig3b_lmd512.dat' using 1:2 title '{/Symbol l} = 512' with lines linetype 2 linewidth 3
unset output

set output 'fig4a.png'
set xlabel 'D_P'
plot [1e-6:1e-1] [-0.02:0.02] 'fig4a_lmd0.1.dat' using 1:2 title '{/Symbol l} = 0.1' with lines linetype 1 linewidth 3,\
'fig4a_lmd1.dat' using 1:2 title '{/Symbol l} = 1' with lines linetype 2 linewidth 3,\
'fig4a_lmd10.dat' using 1:2 title '{/Symbol l} = 10' with lines linetype 3 linewidth 3,\
'fig4a_lmd100.dat' using 1:2 title '{/Symbol l} = 100' with lines linetype 4 linewidth 3
unset output

set output 'fig4b.png'
set xlabel '{/Symbol l}'
plot [1e-1:1e3] [-0.02:0.02] 'fig4b_Dp1e-06.dat' using 1:2 title 'D_P = 10^{-6}' with lines linetype 1 linewidth 3,\
'fig4b_Dp1e-05.dat' using 1:2 title 'D_P = 10^{-5}' with lines linetype 2 linewidth 3,\
'fig4b_Dp0.0001.dat' using 1:2 title 'D_P = 10^{-4}' with lines linetype 3 linewidth 3,\
'fig4b_Dp0.001.dat' using 1:2 title 'D_P = 10^{-3}' with lines linetype 4 linewidth 3
unset output

exit gnuplot
