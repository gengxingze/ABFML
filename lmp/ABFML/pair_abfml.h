#ifdef PAIR_CLASS
PairStyle(abfml, PairABFML)
#else
#ifndef LMP_PAIR_ABFML_H
#define LMP_PAIR_ABFML_H

#include "pair.h"
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace LAMMPS_NS {
class PairABFML : public Pair {
    public:
        explicit PairABFML(class LAMMPS *);
        ~PairABFML() override;
        void settings(int, char **) override;
        void coeff(int, char **) override;
        void init_style() override;
        double init_one(int, int) override;
        void compute(int, int) override;

    private:
        void allocate();
        // Hide Python objects in the implementation to reduce Python-header dependencies.
        struct PythonState;
        std::unique_ptr<PythonState> python;
        std::string model_path;
        std::string device;
        void build_csr();
        // Reuse type mappings and CSR buffers across timesteps to avoid repeated container creation.
        // atomic_numbers: [Nlocal+Nghost]；center_indices: [Nlocal]。
        // edge_ptr: [Nlocal+1]; neighbor_indices: [E]; edge_vectors: flattened [E,3].
        std::vector<int64_t> type_map, atomic_numbers, center_indices, edge_ptr, neighbor_indices;
        std::vector<double> edge_vectors;
        double cutoff = 1.0;
};
}  // namespace LAMMPS_NS
#endif
#endif
