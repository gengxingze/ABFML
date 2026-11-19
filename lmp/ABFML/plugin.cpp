#include "pair_abfml.h"
#include "lammpsplugin.h"
#include "version.h"

using namespace LAMMPS_NS;

namespace {
Pair *factory(LAMMPS *lmp) { return new PairABFML(lmp); }
}

// Dynamic plugin entry point: register abfml for use after plugin load.
extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
    lammpsplugin_t plugin{};
    plugin.version = LAMMPS_VERSION;
    plugin.style = "pair";
    plugin.name = "abfml";
    plugin.info = "ABFML Python runtime pair style";
    plugin.author = "ABFML developers";
    plugin.creator.v1 = reinterpret_cast<lammpsplugin_factory1 *>(&factory);
    plugin.handle = handle;
    auto register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
    register_plugin(&plugin, lmp);
}
