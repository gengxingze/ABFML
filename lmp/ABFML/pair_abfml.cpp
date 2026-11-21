// ABFML Python runtime interface without TorchScript or LibTorch.
// Syntax: pair_style abfml MODEL [DEVICE] [mode eager|compile] [precision float32|float64]
// Each MPI rank owns a Bridge instance and evaluates only local-center energies.
#include "pair_abfml.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "update.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "error.h"
#include "memory.h"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <pybind11/stl.h>
#include <exception>
#include <stdexcept>

namespace py = pybind11;
using namespace LAMMPS_NS;

// Convert elements only when reading pair_coeff, outside the per-step computation.
namespace {

int atomic_number(const std::string &value)
{
    // Array indices are atomic numbers; index 0 is a placeholder, not a real element.
    static const char *const symbols[] = {
        "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };
    for (int number = 1; number <= 118; ++number)
        if (value == symbols[number]) return number;

    throw std::invalid_argument("Invalid element '" + value +
            "': use a case-sensitive element symbol such as Cu");
}

}  // namespace

struct PairABFML::PythonState {
    py::object function;
};

PairABFML::PairABFML(LAMMPS *lmp) : Pair(lmp)
{
    single_enable = 0;
    restartinfo = 0;
    // The model returns global virial directly; disable the unimplemented force-dot-position path.
    no_virial_fdotr_compute = 1;
    centroidstressflag = CENTROID_AVAIL;
}

PairABFML::~PairABFML()
{
    if (python && Py_IsInitialized()) {
        py::gil_scoped_acquire gil;
        python.reset();
    }
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
    }
    // Do not finalize Python in the Pair destructor; other pairs or the host may still use it.
    // Keep self-created interpreters alive until process exit; hold the GIL when destroying Python objects.
}

// Allocate coefficient flags and squared cutoffs for atom-type pairs; index 0 is unused.
void PairABFML::allocate()
{
    allocated = 1;
    const int n = atom->ntypes + 1;
    memory->create(setflag, n, n, "abfml:setflag");
    memory->create(cutsq, n, n, "abfml:cutsq");
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            setflag[i][j] = 0;
            cutsq[i][j] = 0.0;
        }
}

// Parse model path and device; load once and reuse the model for later computations.
void PairABFML::settings(int narg, char **arg)
{
    if (narg < 1)
        error->all(FLERR, "Usage: pair_style abfml MODEL [DEVICE] [mode eager|compile] [precision float32|float64]");
    model_path = arg[0];
    device = "cpu";
    std::string mode = "eager";
    std::string precision = "float32";
    int next = 1;
    // The optional device precedes execution-mode and precision keyword arguments.
    if (next < narg && std::string(arg[next]) != "mode" && std::string(arg[next]) != "precision")
        device = arg[next++];
    bool mode_seen = false, precision_seen = false;
    while (next < narg) {
        const std::string key = arg[next++];
        if (next == narg) error->all(FLERR, "Missing value for ABFML option: {}", key);
        const std::string value = arg[next++];
        if (key == "mode" && !mode_seen) {
            mode_seen = true;
            mode = value;
        } else if (key == "precision" && !precision_seen) {
            precision_seen = true;
            precision = value;
        } else {
            error->all(FLERR, "Unknown or duplicate ABFML option: {}", key);
        }
    }
    // Bare "cuda" selects a GPU by node-local rank; explicit cuda:N stays unchanged.
    int local_rank = 0;
    if (device == "cuda") {
        MPI_Comm node_comm;
        MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, comm->me, MPI_INFO_NULL, &node_comm);
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_free(&node_comm);
    }
    try {
        // Reuse an existing host interpreter when LAMMPS runs from Python; never initialize it twice.
        if (!Py_IsInitialized()) py::initialize_interpreter();
        py::gil_scoped_acquire gil;
        if (device == "cuda") {
            auto cuda = py::module_::import("torch").attr("cuda");
            const int count = cuda.attr("device_count")().cast<int>();
            if (count == 0) throw std::runtime_error("No CUDA devices visible to this MPI rank");
            const int selected = count == 1 ? 0 : local_rank;
            if (selected >= count)
                throw std::runtime_error("More node-local MPI ranks than visible GPUs");
            device = "cuda:" + std::to_string(selected);
            cuda.attr("set_device")(selected);
        }
        auto state = std::make_unique<PythonState>();
        state->function = py::module_::import("abfml.interfaces.lammps").attr("Bridge")(model_path, device, mode, precision);
        cutoff = state->function.attr("cutoff").cast<double>();
        python = std::move(state);
    } catch (const std::exception &e) {
        error->all(FLERR, "ABFML Python load failed: {}", e.what());
    }
}

// Map one-based LAMMPS atom types to atomic numbers and validate model support.
void PairABFML::coeff(int narg, char **arg)
{
    if (narg != atom->ntypes + 2 || std::string(arg[0]) != "*" || std::string(arg[1]) != "*")
        error->all(FLERR, "Usage: pair_coeff * * ELEMENT1 ELEMENT2 ... (element symbols, e.g. Cu)");
    if (!python) error->all(FLERR, "Set pair_style before pair_coeff");
    if (!allocated) allocate();
    try {
        py::gil_scoped_acquire gil;
        auto supported = python->function.attr("type_map").cast<std::vector<int64_t>>();
        type_map.resize(atom->ntypes + 1);
        for (int i = 1; i <= atom->ntypes; ++i) {
            type_map[i] = atomic_number(arg[i+1]);
            if (std::find(supported.begin(), supported.end(), type_map[i]) == supported.end())
                throw std::runtime_error(std::string("Model does not support element '") + arg[i+1] + "'");
            for (int j = i; j <= atom->ntypes; ++j) setflag[i][j] = 1;
        }
    } catch (const std::exception &e) {
        error->all(FLERR, "ABFML coefficients failed: {}", e.what());
    }
}

// Validate units and reverse-force communication, then request full neighbors for local centers.
void PairABFML::init_style()
{
    if (!python) error->all(FLERR, "ABFML Python model is not initialized");
    if (std::string(update->unit_style) != "metal")
        error->all(FLERR, "ABFML Python requires units metal");
    if (!force->newton_pair)
        error->all(FLERR, "ABFML Python requires newton on for ghost-force reverse communication");
    neighbor->add_request(this, NeighConst::REQ_FULL);
}

// Provide pair cutoffs for LAMMPS neighbor construction and communication ranges.
double PairABFML::init_one(int i, int j)
{
    if (!setflag[i][j]) error->all(FLERR, "abfml coefficients are not set");
    return cutoff;
}

// Convert LAMMPS neighbors to CSR: centers are local, neighbors may include ghosts.
void PairABFML::build_csr()
{
    // Local atoms belong to this rank; ghosts copy remote atoms or periodic images.
    const int nlocal = atom->nlocal;
    const int nall = nlocal + atom->nghost;
    double **x = atom->x;
    int *type = atom->type;

    if (list->inum != nlocal)
        error->all(FLERR, "ABFML requires a full neighbor list containing every local atom");

    atomic_numbers.resize(nall);
    for (int i = 0; i < nall; ++i) atomic_numbers[i] = type_map[type[i]];

    center_indices.resize(nlocal);
    edge_ptr.assign(nlocal + 1, 0);
    neighbor_indices.clear();
    edge_vectors.clear();

    const double cutoff2 = cutoff * cutoff;
    for (int slot = 0; slot < list->inum; ++slot) {
        const int i = list->ilist[slot];
        center_indices[slot] = i;
        const int jnum = list->numneigh[i];
        const int *jlist = list->firstneigh[i];

        for (int jj = 0; jj < jnum; ++jj) {
            // Clear special_bonds flags stored in the high bits of neighbor indices.
            const int j = jlist[jj] & NEIGHMASK;
            // Ghost positions already include periodic shifts; subtraction directly gives Rj - Ri.
            const double dx = x[j][0] - x[i][0];
            const double dy = x[j][1] - x[i][1];
            const double dz = x[j][2] - x[i][2];
            const double rsq = dx * dx + dy * dy + dz * dz;
            if (j == i || rsq >= cutoff2) continue;

            neighbor_indices.push_back(j);
            edge_vectors.push_back(dx);
            edge_vectors.push_back(dy);
            edge_vectors.push_back(dz);
        }
        // Edges for this center slot occupy [edge_ptr[slot], edge_ptr[slot + 1]).
        edge_ptr[slot + 1] = static_cast<int64_t>(neighbor_indices.size());
    }
}

// Per-step force entry point: prepare inputs, invoke Python, and accumulate LAMMPS results.
void PairABFML::compute(int eflag, int vflag)
{
    // Initialize energy, atomic-energy, and virial accumulators for this LAMMPS request.
    ev_init(eflag, vflag);
    if (vflag_atom)
        error->all(FLERR, "Use centroid/stress/atom for the nine-component center-assigned virial");
    if (atom->nlocal == 0) return;
    build_csr();
    try {
        py::gil_scoped_acquire gil;
        // Use independently owned NumPy arrays so Python cannot access overwritten C++ buffers next step.
        auto array = [](const std::vector<int64_t> &v) {
            py::array_t<int64_t> result(v.size());
            std::copy(v.begin(), v.end(), result.mutable_data());
            return result;
        };
        py::array_t<double> vectors({static_cast<py::ssize_t>(neighbor_indices.size()), py::ssize_t(3)});
        std::copy(edge_vectors.begin(), edge_vectors.end(), vectors.mutable_data());
        // Receive properties by name and require them only when requested by LAMMPS.
        auto result = python->function.attr("compute")(array(atomic_numbers), array(center_indices),
                array(edge_ptr), array(neighbor_indices), vectors, static_cast<bool>(cvflag_atom)).cast<py::dict>();
        auto required = [&result](const char *name) {
            if (!result.contains(name))
                throw std::runtime_error(std::string("Model does not provide ") + name);
            auto value = py::array_t<double, py::array::c_style>::ensure(result[name]);
            if (!value) throw std::runtime_error(std::string("Invalid output array: ") + name);
            for (py::ssize_t i = 0; i < value.size(); ++i)
                if (!std::isfinite(value.data()[i]))
                    throw std::runtime_error(std::string("Non-finite output: ") + name);
            return value;
        };
        const int nall = atom->nlocal + atom->nghost;
        auto forces = required("forces");
        if (forces.ndim() != 2 || forces.shape(0) != nall || forces.shape(1) != 3)
            throw std::runtime_error("forces must have shape [nlocal + nghost, 3]");

        // Validate every requested output before changing accumulators; missing values must not become zeros.
        double energy = 0.0;
        py::array_t<double, py::array::c_style> ae, w, atomic_w;
        if (eflag_global) {
            auto e = required("energy");
            if (e.size() != 1)
                throw std::runtime_error("energy must contain one local total energy");
            energy = *e.data();
        }
        if (eflag_atom) {
            ae = required("atomic_energies");
            if (ae.ndim() != 1 || ae.shape(0) != atom->nlocal)
                throw std::runtime_error("atomic_energies must have shape [nlocal]");
        }
        if (vflag_global) {
            w = required("virial");
            if (w.ndim() != 2 || w.shape(0) != 3 || w.shape(1) != 3)
                throw std::runtime_error("virial must have shape [3, 3]");
        }
        if (cvflag_atom) {
            atomic_w = required("atomic_virial");
            if (atomic_w.ndim() != 3 || atomic_w.shape(0) != atom->nlocal ||
                    atomic_w.shape(1) != 3 || atomic_w.shape(2) != 3)
                throw std::runtime_error("atomic_virial must have shape [nlocal, 3, 3]");
        }
        auto f = forces.unchecked<2>();
        for (int i = 0; i < nall; ++i)
            for (int d = 0; d < 3; ++d) atom->f[i][d] += f(i, d);
        // LAMMPS reverse-communicates ghost forces with newton on; do not communicate them twice.
        if (eflag_global) eng_vdwl += energy;
        if (eflag_atom) {
            auto e = ae.unchecked<1>();
            for (int slot = 0; slot < atom->nlocal; ++slot)
                eatom[center_indices[slot]] += e(slot);
        }
        // Center assignment: preserve all nine components without symmetry or transposition; add local centers only.
        // Component order: xx yy zz xy xz yz yx zx zy.
        if (cvflag_atom) {
            auto a = atomic_w.unchecked<3>();
            for (int slot = 0; slot < atom->nlocal; ++slot) {
                int i = center_indices[slot];
                cvatom[i][0] += a(slot, 0, 0);
                cvatom[i][1] += a(slot, 1, 1);
                cvatom[i][2] += a(slot, 2, 2);
                cvatom[i][3] += a(slot, 0, 1);
                cvatom[i][4] += a(slot, 0, 2);
                cvatom[i][5] += a(slot, 1, 2);
                cvatom[i][6] += a(slot, 1, 0);
                cvatom[i][7] += a(slot, 2, 0);
                cvatom[i][8] += a(slot, 2, 1);
            }
        }
        // Virial is not divided by volume; six-component order: xx yy zz xy xz yz.
        if (vflag_global) {
            auto v = w.unchecked<2>();
            virial[0] += v(0, 0); virial[1] += v(1, 1); virial[2] += v(2, 2);
            // Global six-component virial uses the symmetric tensor part; atomic nine-component tensors remain unchanged.
            virial[3] += 0.5 * (v(0, 1) + v(1, 0));
            virial[4] += 0.5 * (v(0, 2) + v(2, 0));
            virial[5] += 0.5 * (v(1, 2) + v(2, 1));
        }
    } catch (const std::exception &e) {
        error->all(FLERR, "ABFML Python compute failed: {}", e.what());
    }
}

