from abfml.calculate import ABFML
from ase.build import bulk

calc = ABFML('D:\Work\PyCharm\ABFML\example\DP-sea\model.pt')
structure = bulk('Cu', a=3.62)
structure.calc = calc
energy = structure.get_potential_energy()
force = structure.get_forces()