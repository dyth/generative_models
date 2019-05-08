import pstats
p = pstats.Stats('out')
p.strip_dirs().sort_stats('cumulative').print_stats(50)
