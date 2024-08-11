#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>

#include "pair_ABFML.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "domain.h"


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairABFML::PairABFML(LAMMPS* lmp) : Pair(lmp)
{
    writedata = 1;
}

PairABFML::~PairABFML()
{
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
    }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairABFML::allocate()
{
    allocated = 1;
    int np1 = atom->ntypes;
    memory->create(setflag, np1 + 1, np1 + 1, "pair:setflag");
    for (int i = 1; i <= np1; i++)
        for (int j = i; j <= np1; j++) setflag[i][j] = 0;
    memory->create(cutsq, np1 + 1, np1 + 1, "pair:cutsq");

}

static bool is_key(const std::string& input) {
    std::vector<std::string> keys;
    keys.push_back("out_freq");
    keys.push_back("out_file");

    for (int ii = 0; ii < keys.size(); ++ii) {
        if (input == keys[ii]) {
            return true;
        }
    }
    return false;
}

static int stringCmp(const void* a, const void* b) {
    char* m = (char*)a;
    char* n = (char*)b;
    int i, sum = 0;

    for (i = 0; i < MPI_MAX_PROCESSOR_NAME; i++) {
        if (m[i] == n[i]) {
            continue;
        }
        else {
            sum = m[i] - n[i];
            break;
        }
    }
    return sum;
}

int PairABFML::get_node_rank() {
    char host_name[MPI_MAX_PROCESSOR_NAME];
    memset(host_name, '\0', sizeof(char) * MPI_MAX_PROCESSOR_NAME);
    char(*host_names)[MPI_MAX_PROCESSOR_NAME];
    int n, namelen, color, rank, nprocs, myrank;
    size_t bytes;
    MPI_Comm nodeComm;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Get_processor_name(host_name, &namelen);

    bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
    host_names = (char(*)[MPI_MAX_PROCESSOR_NAME])malloc(bytes);
    for (int ii = 0; ii < nprocs; ii++) {
        memset(host_names[ii], '\0', sizeof(char) * MPI_MAX_PROCESSOR_NAME);
    }

    strcpy(host_names[rank], host_name);

    for (n = 0; n < nprocs; n++) {
        MPI_Bcast(&(host_names[n]), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n,
            MPI_COMM_WORLD);
    }
    qsort(host_names, nprocs, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

    color = 0;
    for (n = 0; n < nprocs - 1; n++) {
        if (strcmp(host_name, host_names[n]) == 0) {
            break;
        }
        if (strcmp(host_names[n], host_names[n + 1])) {
            color++;
        }
    }

    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);
    MPI_Comm_rank(nodeComm, &myrank);

    MPI_Barrier(MPI_COMM_WORLD);
    int looprank = myrank;
    // printf (" Assigning device %d  to process on node %s rank %d,
    // OK\n",looprank,  host_name, rank );
    free(host_names);
    return looprank;
}



/* ----------------------------------------------------------------------
   global settings pair_style
------------------------------------------------------------------------- */

void PairABFML::settings(int narg, char** arg)
{
    if (narg <= 0) error->all(FLERR, "Illegal pair_style command");
    std::vector<std::string> arg_vector;

    int iarg;
    while (iarg < narg) {
        if (is_key(arg[iarg])) {
            break;
        }
        iarg++;
    }

    for (int ii = 0; ii < iarg; ++ii) {
        arg_vector.push_back(arg[ii]);
    }

    if (arg_vector.size() == 1) {
        try
        {
            std::string model_file = arg_vector[0];
            torch::jit::getExecutorMode() = false;
            model = torch::jit::load(model_file);
            if (torch::cuda::is_available()) { device = torch::Device(torch::kCUDA, get_node_rank());}
            if (true) { dtype = torch::kFloat32; }
            model.to(dtype);
            model.to(device);
            model.eval();
            if (comm->me == 0)
            {
                utils::logmesg(lmp, "Load model successful !----> %s", model_file);
                utils::logmesg(lmp, "INFO IN ABFML-MODEL---->>");
            }

        }
        catch (const c10::Error e)
        {
            std::cerr << "Failed to load model!" << e.msg() << std::endl;
        }
    }

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs pair_coeff
------------------------------------------------------------------------- */

void PairABFML::coeff(int narg, char** arg)
{
    int ntype = atom->ntypes;
    if (!allocated) { allocate(); }

    // pair_coeff * * 
    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
        for (int j = MAX(jlo, i); j <= jhi; j++)
        {
            setflag[i][j] = 1;
            count++;
        }
    }
    cutoff = model.attr("cutoff").toDouble();
    std::vector<int64_t> neighbor_model = model.attr("neighbor").toIntVector();
    auto type_map_model = model.attr("type_map").toList();
    if (ntype > narg - 2)
    {
        error->all(FLERR, "Element mapping not fully set");
    }
    for (int ii = 2; ii < narg; ++ii) {
        int temp = std::stoi(arg[ii]);
        auto iter = std::find(type_map_model.begin(), type_map_model.end(), temp);
        if (iter != type_map_model.end() || arg[ii] == 0)
        {
            type_map.push_back(temp);
        }
        else
        {
            error->all(FLERR, "This element is not included in the machine learning force field");
        }
    }
    if (neighbor_model.size() == 1)
    {
        max_neighbor = neighbor_model[0];
    }
    else
    {
        std::vector<int> type_map_temp = type_map;
        std::sort(type_map_temp.begin(), type_map_temp.end());
        for (int ii = 0; ii < type_map.size(); ii++)
        {
            neighbor_map.push_back(neighbor_model[std::find(type_map_model.begin(), type_map_model.end(), type_map_temp[ii]) - type_map_model.begin()]);
            max_neighbor += neighbor_map[ii];
            neighbor_width.push_back(max_neighbor);
        }
    }
    
    if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairABFML::init_one(int i, int j)
{
    //if (setflag[i][j] == 0) { error->all(FLERR, "All pair coeffs are not set"); 
    return cutoff;
}


void PairABFML::init_style()
{
    // Using a nearest neighbor table of type full
    neighbor->add_request(this, NeighConst::REQ_FULL);
}
/* ---------------------------------------------------------------------- */


void PairABFML::calculate_neighbor()
{
    double** x = atom->x;
    int* type = atom->type;
    int nlocal = atom->nlocal;
    int n_all = nlocal + atom->nghost;
    int* ilist, * jlist, * numneigh, ** firstneigh;
    int inum, jnum, itype, jtype;
    double dx, dy, dz, rsq, rij;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    image_type.clear();
    image_type.resize(nlocal, 0);
    neighbor_type.clear();
    neighbor_type.resize(nlocal * max_neighbor, -1);
    neighbor_list.clear();
    neighbor_list.resize(nlocal * max_neighbor, -1);
    image_dR.clear();
    image_dR.resize(nlocal * max_neighbor * 4, 0);


    std::vector<int> use_type(n_all);

    for (int ii = 0; ii < n_all; ii++)
    {
        use_type[ii] = type_map[type[ii] - 1];
    }

    double rc2 = cutoff * cutoff;

    if (neighbor_map.size() == 0)
    {
        for (int ii = 0; ii < inum; ii++)
        {
            int i = ilist[ii];
            jlist = firstneigh[i];
            image_type[ii] = use_type[i];
            int kk = 0;
            for (int jj = 0; jj < numneigh[i]; jj++)
            {
                int j = jlist[jj];
                dx = x[j][0] - x[i][0];
                dy = x[j][1] - x[i][1];
                dz = x[j][2] - x[i][2];
                rsq = dx * dx + dy * dy + dz * dz;
                if (rsq <= rc2) {
                    rij = sqrt(rsq);
                    int indices = ii * max_neighbor * 4 + kk * 4;
                    image_dR[indices] = rij;
                    image_dR[indices + 1] = dx;
                    image_dR[indices + 2] = dy;
                    image_dR[indices + 3] = dz;
                    neighbor_list[ii * max_neighbor + kk] = j;
                    neighbor_type[ii * max_neighbor + kk] = use_type[j];
                    kk = kk + 1;
                }
            }
            if (kk > max_neighbor)
                {
                    std::vector<std::tuple<double, double, double, double, int, int>> data;
                    data.resize(kk);
                    int indices = ii * max_neighbor * 4;
                    for (int mm = 0; mm <= kk; mm++) {
                        data[mm] = std::make_tuple(image_dR[mm * 4],
                            image_dR[mm * 4 + 1],
                            image_dR[mm * 4 + 2],
                            image_dR[mm * 4 + 3],
                            neighbor_list[ii * max_neighbor + mm],
                            neighbor_type[ii * max_neighbor + mm]);
                    }
                    std::sort(data.begin(), data.end(), [](const auto& lhs, const auto& rhs) {
                        return std::get<0>(lhs) > std::get<0>(rhs);
                        });
                    for (int mm = 0; mm < max_neighbor; mm++)
                    {
                        image_dR[indices] = std::get<0>(data[mm]);
                        image_dR[indices + 1] = std::get<1>(data[mm]);
                        image_dR[indices + 2] = std::get<2>(data[mm]);
                        image_dR[indices + 3] = std::get<3>(data[mm]);
                        neighbor_list[ii * max_neighbor + mm] = std::get<4>(data[mm]);
                        neighbor_type[ii * max_neighbor + mm] = std::get<5>(data[mm]);
                    }
                }
        }
    }
    else
    {
        for (int ii = 0; ii < inum; ii++)
        {
            int i = ilist[ii];
            jlist = firstneigh[i];
            image_type[ii] = use_type[i];
            std::vector<int> kk(type_map.size(), 0);
            for (int jj = 0; jj < numneigh[i]; jj++)
            {
                int j = jlist[jj];
                dx = x[j][0] - x[i][0];
                dy = x[j][1] - x[i][1];
                dz = x[j][2] - x[i][2];
                rsq = dx * dx + dy * dy + dz * dz;
                if (rsq <= rc2) {
                    rij = sqrt(rsq);
                    jtype = type[j];
                    int indices_1 = (ii * max_neighbor + neighbor_width[jtype - 1] + kk[jtype - 1]) * 4;
                    int indices_2 = ii * max_neighbor + neighbor_width[jtype - 1] + kk[jtype - 1];
                    image_dR[indices_1] = rij;
                    image_dR[indices_1 + 1] = dx;
                    image_dR[indices_1 + 2] = dy;
                    image_dR[indices_1 + 3] = dz;
                    neighbor_list[indices_2] = j;
                    neighbor_type[indices_2] = use_type[j];
                    kk[jtype-1] = kk[jtype-1] + 1;   
                }
            }

        }
    }
}


void PairABFML::compute(int eflag, int vflag)
{

    auto t1 = std::chrono::high_resolution_clock::now();
    if (eflag || vflag) ev_setup(eflag, vflag);
    double** f = atom->f;
    int newton_pair = force->newton_pair;
    int nlocal = atom->nlocal;
    int nghost =  nghost = atom->nghost; 
    int n_all = nlocal + nghost;

    auto t2 = std::chrono::high_resolution_clock::now();
    torch::Tensor element_tensor = torch::zeros({ type_map.size() }, torch::kInt);
    for (int i = 0; i < type_map.size(); i++)
    {
        element_tensor[i] = type_map[i];
    }
    calculate_neighbor();
    if (nlocal > 0)
    {

        torch::Tensor image_type_tensor = torch::from_blob(image_type.data(), { 1,nlocal }, torch::TensorOptions().dtype(torch::kInt)).to(device);
        torch::Tensor neighbor_list_tensor = torch::from_blob(neighbor_list.data(), { 1,nlocal, max_neighbor }, torch::TensorOptions().dtype(torch::kInt)).to(device);
        torch::Tensor neighbor_type_tensor = torch::from_blob(neighbor_type.data(), { 1,nlocal, max_neighbor }, torch::TensorOptions().dtype(torch::kInt)).to(device);
        torch::Tensor image_dR_tensor = torch::from_blob(image_dR.data(), { 1, nlocal, max_neighbor, 4 }, torch::TensorOptions().dtype(torch::kFloat64)).to(device, dtype);

        auto output = model.forward({ image_type_tensor, neighbor_list_tensor, neighbor_type_tensor, image_dR_tensor, nghost }).toTuple();

        torch::Tensor Etot = output->elements()[0].toTensor().to(torch::kCPU, torch::kFloat64);
        torch::Tensor Ei = output->elements()[1].toTensor().to(torch::kCPU, torch::kFloat64);
        torch::Tensor Force = output->elements()[2].toTensor().to(torch::kCPU, torch::kFloat64);
        torch::Tensor Virial = output->elements()[3].toTensor().to(torch::kCPU, torch::kFloat64);
        torch::Tensor Virial_atoms = output->elements()[4].toTensor().to(torch::kCPU, torch::kFloat64);
        // get force
        auto force_ptr = Force.accessor<double, 3>();
        auto Ei_ptr = Ei.accessor <double, 3>();
        auto virial_ptr = Virial.accessor<double, 2>();
        auto virial_atoms_ptr = Virial_atoms.accessor<double, 3>();


        for (int i = 0; i < n_all; i++)
        {
            f[i][0] += force_ptr[0][i][0];
            f[i][1] += force_ptr[0][i][1];
            f[i][2] += force_ptr[0][i][2];
        }

        // get energy
        if (eflag)  eng_vdwl = Etot[0][0].item<double>();

        if (eflag_atom)
        {
            for (int ii = 0; ii < nlocal; ii++)
            {
                eatom[ii] = Ei_ptr[0][ii][0];
            }
        }
        if (vflag_atom) {
            error->all(FLERR,
                "6-element atomic virial is not supported. Use compute "
                "centroid/stress/atom command for 9-element atomic virial.");
        }
        if (cvflag_atom) {
            for (int ii = 0; ii < n_all; ++ii) {
                cvatom[ii][0] += virial_atoms_ptr[0][ii][0];  // xx
                cvatom[ii][1] += virial_atoms_ptr[0][ii][4];  // yy
                cvatom[ii][2] += virial_atoms_ptr[0][ii][8];  // zz
                cvatom[ii][3] += virial_atoms_ptr[0][ii][3];  // xy
                cvatom[ii][4] += virial_atoms_ptr[0][ii][6];  // xz
                cvatom[ii][5] += virial_atoms_ptr[0][ii][7];  // yz
                cvatom[ii][6] += virial_atoms_ptr[0][ii][1];  // yx
                cvatom[ii][7] += virial_atoms_ptr[0][ii][2];  // zx
                cvatom[ii][8] += virial_atoms_ptr[0][ii][5];  // zy
            }
        }
        if (vflag) {
            virial[0] = virial_ptr[0][0];    // xx
            virial[1] = virial_ptr[0][4];    // yy
            virial[2] = virial_ptr[0][8];    // zz
            virial[3] = virial_ptr[0][1];    // xy
            virial[4] = virial_ptr[0][2];    // xz
            virial[5] = virial_ptr[0][5];    // yz
        }
    }
    

}
