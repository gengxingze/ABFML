// SPDX-License-Identifier: GPL-3.0
/**
 * ABFML plugin for LAMMPS
 * See https://docs.lammps.org/Developer_plugins.html
 */

 #include "pair_abfml.h"
 #include "lammpsplugin.h"
 #include "version.h"

 using namespace LAMMPS_NS;

 static Pair *pairabfml(LAMMPS *lmp) { return new PairABFML(lmp); }

 extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
   lammpsplugin_t plugin;
   lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc)regfunc;

   plugin.version = LAMMPS_VERSION;
   plugin.style = "pair";
   plugin.name = "abfml";
   plugin.info = "A problem-oriented package for rapidly creating, screening, and optimizing new machine learning force fields";
   plugin.author = "Geng Xingze";
   plugin.creator.v1 = (lammpsplugin_factory1 *)&pairabfml;
   plugin.handle = handle;
   (*register_plugin)(&plugin, lmp);
 }
 