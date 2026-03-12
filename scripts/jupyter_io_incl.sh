#!/bin/bash
# Jupiter/Io inclination sweep 
python plot_incl.py \
    --orbits 1000 \
    --divisor 2 \
    --window 48 \
    --a-p 7.78e11 \
    --M-p 1.989e30 \
    --r-p 7.15e7 \
    --a-m 4.22e8 \
    --M-m 1.898e27 \
    --name jupyter_io_incl
