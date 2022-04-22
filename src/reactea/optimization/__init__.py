

try:
    from reactea.optimization.jmetal.ea import EA as JMetalEA
    engine = JMetalEA
except ImportError:
    print("jmetal not available")

algorithms = ['SA', 'GA', 'NSGAII', 'SPEA2', 'NSGAIII', 'GDE3']

preferred_EA = 'NSGAII'



