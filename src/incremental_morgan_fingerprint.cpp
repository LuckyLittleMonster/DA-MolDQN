#include "incremental_morgan_fingerprint.h"

// #define CENV_DEBUG


using namespace RDKit;
namespace python = boost::python;
namespace numpy = boost::python::numpy;

using std::cout;
using std::endl;

// typedef std::tuple<boost::dynamic_bitset<>, unsigned int, unsigned int> MyAccumTuple;
using MyAccumTuple = boost::tuple<boost::dynamic_bitset<>, unsigned int, unsigned int>;

void initialize_numpy()
{
    static bool f = true;
    if (f) {
        Py_Initialize();
        numpy::initialize();
        import_array();
        f = false;    
    }
    
}

IncrementalMorganFingerprint::IncrementalMorganFingerprint() {
    hash_value_table.resize(radius + 1, Map_T());
    fpGenerator = RDKit::MorganFingerprint::getMorganGenerator<std::uint32_t>(radius);
    initialize_numpy();
}

IncrementalMorganFingerprint::IncrementalMorganFingerprint(uint radius, uint length) {
    this->radius = radius;
    this->length = length;
    hash_value_table.resize(radius + 1, Map_T());
    fpGenerator = RDKit::MorganFingerprint::getMorganGenerator<std::uint32_t>(radius);
    initialize_numpy();
}

void IncrementalMorganFingerprint::setBaseMol(const ROMol &mol) {

    this->baseMol = mol;
    auto na = mol.getNumAtoms();
    for (auto &hvt : hash_value_table) {
        hvt.resize(na);
    }

    this->calcFingerprint(mol, radius, nullptr, nullptr, useChirality, useBondTypes, onlyNonzeroInvariants, includeRedundantEnvironments, hash_value_table);

    bond_addition_hvt = hash_value_table;
    for (auto &hvt : bond_addition_hvt) {
        for (auto &val : hvt) {
            val.first.push_back(0);
        }
    }

    atom_addition_hvt = bond_addition_hvt;
    for (auto &hvt : atom_addition_hvt) {
        hvt.resize(na + 1);
    }
}

IMFCache IncrementalMorganFingerprint::cacheState() const {
    return {hash_value_table, bond_addition_hvt, atom_addition_hvt, baseMol, invariants};
}

void IncrementalMorganFingerprint::restoreState(const IMFCache &cache) {
    hash_value_table = cache.hash_value_table;
    bond_addition_hvt = cache.bond_addition_hvt;
    atom_addition_hvt = cache.atom_addition_hvt;
    baseMol = cache.baseMol;
    invariants = cache.invariants;
}

boost::python::list IncrementalMorganFingerprint::getBaseMolMorganFingerprint() {
    return convertTableToSparseList(hash_value_table);
}
boost::python::object IncrementalMorganFingerprint::getBaseMolMorganFingerprintAsNumPy() {
    return convertTableToNumPy(hash_value_table);
}

boost::python::list IncrementalMorganFingerprint::getIncrementalMorganFingerprint(const RDKit::ROMol &new_mol, const std::list<int> &fromAtoms) {
    // Table_T new_hash_value_table(hash_value_table);
    // this->calcIncrementalFingerprint(new_mol, fromAtoms, new_hash_value_table);
    Table_T new_hash_value_table;
    if (new_mol.getNumAtoms() > this->baseMol.getNumAtoms()) {
        // atom addition
        new_hash_value_table = this->atom_addition_hvt;
    } else if (new_mol.getNumBonds() > this->baseMol.getNumBonds()) {
        // bond addition
        new_hash_value_table = this->bond_addition_hvt;
    } else {
        new_hash_value_table = this->hash_value_table;
    }
    this->calcIncrementalFingerprint(new_mol, fromAtoms, new_hash_value_table);
    return convertTableToSparseList(new_hash_value_table);
}


bool equal(std::set<uint32_t> &s1, std::set<uint32_t> &s2) {
    if (s1.size() != s2.size()) {
        return false;
    }

    auto it = s1.begin();
    auto jt = s2.begin();

    while(it != s1.end()) {
        if (*it != *jt) {
            return false;
        }
        ++it;
        ++jt;
    }
    return true;
}


boost::python::object IncrementalMorganFingerprint::getIncrementalMorganFingerprintAsNumPy(const RDKit::ROMol &new_mol, const std::list<int> &fromAtoms) {

    Table_T new_hash_value_table;
    if (new_mol.getNumAtoms() > this->baseMol.getNumAtoms()) {
        // atom addition
        new_hash_value_table = this->atom_addition_hvt;
    } else if (new_mol.getNumBonds() > this->baseMol.getNumBonds()) {
        // bond addition
        new_hash_value_table = this->bond_addition_hvt;
    } else {
        new_hash_value_table = this->hash_value_table;
    }

    this->calcIncrementalFingerprint(new_mol, fromAtoms, new_hash_value_table);
#ifdef CENV_DEBUG
    auto s = convertTableToSet(new_hash_value_table);
    RDKit::MorganFingerprints::BitInfoMap *bitInfoMap = new RDKit::MorganFingerprints::BitInfoMap();
    auto v = RDKit::MorganFingerprints::getFingerprint(new_mol, radius, nullptr, nullptr, useChirality, useBondTypes, true, onlyNonzeroInvariants, bitInfoMap, 0);
    // auto v = debug_getFingerprint(new_mol, radius, nullptr, nullptr, useChirality, useBondTypes, true, onlyNonzeroInvariants, bitInfoMap, 0);
    std::set<uint32_t> vs;
    for (auto &val : v->getNonzeroElements()) {
        // cout << val.first % this->length << ", ";
        vs.insert(val.first);
    }
    
    if (!equal(s, vs)) 
    {

        cout << "baseMol: " << MolToSmiles(this->baseMol) << endl;
        // for (auto atom : this->baseMol.atoms()) {
        //     auto atomic_num = atom->getAtomicNum();
        //     auto atomic_idx = atom->getIdx();
        //     cout << atomic_idx << ": ";
        //     cout << 
        // }
        {
            uint nAtoms = this->baseMol.getNumAtoms();
            cout << "baseMol hash_value_table: " << endl;
            for (uint i = 0; i < nAtoms; ++i) {
                cout << i << " : ";
                for (uint j = 0; j < (radius + 1); ++j) {
                    cout << "{" 
                        // << this->hash_value_table[j][i].first << "," 
                        << this->hash_value_table[j][i].second << "},";
                }
                cout << endl;
            }
        }
        

        cout << "new_mol: " << MolToSmiles(new_mol) << endl;
        cout << "s: ";
        for (auto &val : s) {
            cout << val << ", ";
        }
        cout << endl;

        uint nAtoms = new_mol.getNumAtoms();
        cout << "new hash_value_table: " << endl;
        for (uint i = 0; i < nAtoms; ++i) {
            cout << i << " : ";
            for (uint j = 0; j < (radius + 1); ++j) {
                cout << "{" 
                    // << new_hash_value_table[j][i].first << "," 
                    << new_hash_value_table[j][i].second << "},";
            }
            cout << endl;
        }
        
        cout << "v: ";
        for (auto &val : vs) {
            cout << val << ", ";
        }
        cout << endl;

        cout << "bitInfoMap: ";
        for (auto &l : *bitInfoMap) {
            cout << "{" << l.first << ": ";
            for (auto &p : l.second) {
                cout << "(" << p.first << "," << p.second << "),";
            }
            cout << "},";
        }
        cout << endl;

        auto na = new_mol.getNumAtoms();
        Table_T debug_table(this->radius + 1);
        for (auto &hvt : debug_table) {
            hvt.resize(na);
        }
        this->calcFingerprint(new_mol, radius, nullptr, nullptr, useChirality, useBondTypes, onlyNonzeroInvariants, includeRedundantEnvironments, debug_table);
        cout << "debug hash_value_table: " << endl;
        for (uint i = 0; i < nAtoms; ++i) {
            cout << i << " : ";
            for (uint j = 0; j < (radius + 1); ++j) {
                cout << "{" 
                    // << debug_table[j][i].first << "," 
                    << debug_table[j][i].second << "},";
            }
            cout << endl;
        }
    }
    delete bitInfoMap;
#endif
    return convertTableToNumPy(new_hash_value_table);
}


boost::python::list IncrementalMorganFingerprint::getMorganFingerprint(const RDKit::ROMol &new_mol) {
    auto v = RDKit::MorganFingerprints::getFingerprint(new_mol, radius, nullptr, nullptr, useChirality, useBondTypes, true, onlyNonzeroInvariants, nullptr, 0);
    boost::python::list lt;
    for (auto &val : v->getNonzeroElements()) {
        lt.append(val.first % this->length);
    }
    return lt;
}

boost::python::object IncrementalMorganFingerprint::getMorganFingerprintAsNumPy(const RDKit::ROMol &new_mol) {
    auto v = RDKit::MorganFingerprints::getFingerprint(new_mol, radius, nullptr, nullptr, useChirality, useBondTypes, true, onlyNonzeroInvariants, nullptr, 0);
    npy_intp size[1] = {static_cast<npy_intp>(this->length)};

    PyObject *arr = PyArray_ZEROS(1, size, NPY_UINT8, 0);
    PyObject *one = PyLong_FromLong(1);

    for (auto &val : v->getNonzeroElements()) {
        PyArray_SETITEM(
            (PyArrayObject *)arr,
            static_cast<char *>(PyArray_GETPTR1((PyArrayObject *)arr, val.first % this->length)), one);
    }

    Py_DECREF(one);
    python::handle<> res(arr);
    return python::object(res);
}

boost::python::list IncrementalMorganFingerprint::getMorganFingerprintByGenerator(const RDKit::ROMol &new_mol) {
    // todo
    boost::python::list lt;
    return lt;
}

boost::python::object IncrementalMorganFingerprint::getMorganFingerprintAsNumPyByGenerator(const RDKit::ROMol &new_mol) {
    ExplicitBitVect *result = fpGenerator->getFingerprint(
            new_mol, nullptr, nullptr, -1, nullptr, nullptr, nullptr);
        // mol, fromAtoms, ignoreAtoms, confId, additionalOutput,
        // customAtomInvariants, customBondInvariants);

    npy_intp size[1] = {static_cast<npy_intp>(result->size())};
    PyObject *arr = PyArray_ZEROS(1, size, NPY_UINT8, 0);
    PyObject *one = PyLong_FromLong(1);
    for (auto i = 0u; i < result->size(); ++i) {
      if ((*result)[i]) {
        PyArray_SETITEM(
            (PyArrayObject *)arr,
            static_cast<char *>(PyArray_GETPTR1((PyArrayObject *)arr, i)), one);
      }
    }
    Py_DECREF(one);
    python::handle<> res(arr);
    return python::object(res);
}

// Pure C++ dense FP methods — no Python objects, safe without GIL

std::vector<uint8_t> IncrementalMorganFingerprint::convertTableToDenseVector(Table_T &table) {
    auto s = convertTableToSet(table);
    std::vector<uint8_t> dense(this->length, 0);
    for (auto &val : s) {
        dense[val % this->length] = 1;
    }
    return dense;
}

std::vector<uint8_t> IncrementalMorganFingerprint::getBaseMolMorganFingerprintDense() {
    return convertTableToDenseVector(hash_value_table);
}

std::vector<uint8_t> IncrementalMorganFingerprint::getIncrementalMorganFingerprintDense(const RDKit::ROMol &new_mol, const std::list<int> &fromAtoms) {
    Table_T new_hash_value_table;
    if (new_mol.getNumAtoms() > this->baseMol.getNumAtoms()) {
        new_hash_value_table = this->atom_addition_hvt;
    } else if (new_mol.getNumBonds() > this->baseMol.getNumBonds()) {
        new_hash_value_table = this->bond_addition_hvt;
    } else {
        new_hash_value_table = this->hash_value_table;
    }
    this->calcIncrementalFingerprint(new_mol, fromAtoms, new_hash_value_table);
    return convertTableToDenseVector(new_hash_value_table);
}

std::vector<uint8_t> IncrementalMorganFingerprint::getMorganFingerprintDense(const RDKit::ROMol &new_mol) {
    auto v = RDKit::MorganFingerprints::getFingerprint(new_mol, radius, nullptr, nullptr, useChirality, useBondTypes, true, onlyNonzeroInvariants, nullptr, 0);
    std::vector<uint8_t> dense(this->length, 0);
    for (auto &val : v->getNonzeroElements()) {
        dense[val.first % this->length] = 1;
    }
    delete v;
    return dense;
}


// std::set<uint32_t> IncrementalMorganFingerprint::convertTableToSet(Table_T &table) {
//     std::set<uint32_t> s;
//     std::vector<boost::dynamic_bitset<>> neighborhoods;

//     auto &l = table[0];
//     boost::dynamic_bitset<> deadAtoms(l.size());
//     for (auto &val : l) {

//         auto &nb = val.first;
//         auto &el = val.second;

//         neighborhoods.push_back(nb);
//         s.insert(el);
//     }

//     for (size_t i = 1; i < table.size(); ++i) {
//         std::vector<RDKit::MorganFingerprints::MyAccumTuple> neighborhoodsThisRound;
//         auto &l = table[i];
//         for (uint j = 0; j < l.size(); ++j) {
//             auto &atomIdx = j;
//             auto &val = l[j];
//             auto &nb = val.first;
//             auto &el = val.second;

//             if (std::find(neighborhoods.begin(), neighborhoods.end(), nb) == neighborhoods.end()) {
//                 neighborhoods.push_back(nb);
//                 s.insert(el);
//             }
//         }
//     }
//     return s;
// }

// namespace std {

//     template <typename Block, typename Alloc> struct hash<boost::dynamic_bitset<Block, Alloc> > {
//         size_t operator()(boost::dynamic_bitset<Block, Alloc> const& bs) const {
//             size_t seed = boost::hash_value(bs.size());

//             std::vector<Block> blocks(bs.num_blocks());
//             boost::hash_range(seed, blocks.begin(), blocks.end());

//             return seed;
//         }
//     };

// }

std::set<uint32_t> IncrementalMorganFingerprint::convertTableToSet(Table_T &table) {
    std::set<uint32_t> s;
    // std::vector<boost::dynamic_bitset<>> neighborhoods;
    std::unordered_set<boost::dynamic_bitset<>> neighborhoods;

    auto &l = table[0];
    boost::dynamic_bitset<> deadAtoms(l.size());
    for (auto &val : l) {

        auto &nb = val.first;
        auto &el = val.second;

        // neighborhoods.push_back(nb);
        neighborhoods.insert(nb);
        s.insert(el);
    }

    for (size_t i = 1; i < table.size(); ++i) {
        std::vector<MyAccumTuple> neighborhoodsThisRound;
        auto &l = table[i];
        for (uint j = 0; j < l.size(); ++j) {
            auto &atomIdx = j;
            auto &val = l[j];
            auto &nb = val.first;
            auto &el = val.second;

            if (!deadAtoms[atomIdx]) {
                // neighborhoodsThisRound.push_back(boost::make_tuple(nb, el, atomIdx));
                neighborhoodsThisRound.push_back(MyAccumTuple(nb, el, atomIdx));
            }
        }
        std::sort(neighborhoodsThisRound.begin(), neighborhoodsThisRound.end());

        for (std::vector<MyAccumTuple>::const_iterator iter =
                         neighborhoodsThisRound.begin();
                 iter != neighborhoodsThisRound.end(); ++iter) {
            if (
                // std::find(neighborhoods.begin(), neighborhoods.end(), iter->get<0>()) == neighborhoods.end()
                neighborhoods.find(iter->get<0>()) == neighborhoods.end()
                ) {
                // neighborhoods.push_back(iter->get<0>());
                neighborhoods.insert(iter->get<0>());
                s.insert(iter->get<1>());
            } else {
                deadAtoms[iter->get<2>()] = 1;
            }
        }
    }


    #ifdef CENV_DEBUG
    cout << "set: ";
    for (auto &val : s) {
        cout << val << ",";
    }
    cout << endl;
    #endif

    return s;
}

boost::python::list IncrementalMorganFingerprint::convertTableToSparseList(Table_T &table) {
    auto s = convertTableToSet(table);
    boost::python::list lt;
    for (auto &val : s) {
        lt.append(val % this->length);
    }

    return lt;
}

boost::python::object IncrementalMorganFingerprint::convertTableToNumPy(Table_T &table) {
    #ifdef CENV_DEBUG
    std::cout << "convertTableToNumPy start" << endl;
    #endif
    auto s = convertTableToSet(table);

    npy_intp size[1] = {static_cast<npy_intp>(this->length)};

    PyObject *arr = PyArray_ZEROS(1, size, NPY_UINT8, 0);
    PyObject *one = PyLong_FromLong(1);

    for (auto &val : s) {
        PyArray_SETITEM(
            (PyArrayObject *)arr,
            static_cast<char *>(PyArray_GETPTR1((PyArrayObject *)arr, val % this->length)), one);
    }

    Py_DECREF(one);
    python::handle<> res(arr);

    #ifdef CENV_DEBUG
    std::cout << "convertTableToNumPy done" << endl;
    #endif
    return python::object(res);
}


void IncrementalMorganFingerprint::calcFingerprint(const ROMol &mol, uint radius,
    std::vector<uint32_t> *invariants,
    const std::set<uint32_t> *fromAtoms, bool useChirality,
    bool useBondTypes, 
    bool onlyNonzeroInvariants,
    bool includeRedundantEnvironments,
    Table_T &res
    ) 
{
    using namespace RDKit::MorganFingerprints;
    #if 0
    std::cout << "calcFingerprint" << std::endl;
    std::cout << "useChirality: " << useChirality << std::endl;
    std::cout << "useBondTypes: " << useBondTypes << std::endl; 
    std::cout << "onlyNonzeroInvariants: " << onlyNonzeroInvariants << std::endl; 
    std::cout << "includeRedundantEnvironments: " << includeRedundantEnvironments << std::endl;
    #endif
    const ROMol *lmol = &mol;
    std::unique_ptr<ROMol> tmol;
    if (useChirality && !mol.hasProp(common_properties::_StereochemDone)) {
        tmol = std::unique_ptr<ROMol>(new ROMol(mol));
        MolOps::assignStereochemistry(*tmol);
        lmol = tmol.get();
    }
    uint nAtoms = lmol->getNumAtoms();
    bool owner = false;
    if (!invariants) {
        invariants = new std::vector<uint32_t>(nAtoms);
        owner = true;
        getConnectivityInvariants(*lmol, *invariants);
    }
    // Make a copy of the invariants:
    std::vector<uint32_t> invariantCpy(nAtoms);
    std::copy(invariants->begin(), invariants->end(), invariantCpy.begin());
    this->invariants.resize(nAtoms);
    std::copy(invariants->begin(), invariants->end(), this->invariants.begin());


    // add the round 0 invariants to the result:
    boost::dynamic_bitset<> empty_nb(mol.getNumBonds());
    for (uint i = 0; i < nAtoms; ++i) {
        if (!fromAtoms || fromAtoms->find(i) != fromAtoms->end()) {
            if (!onlyNonzeroInvariants || (*invariants)[i]) {
                // uint32_t bit = updateElement(res, (*invariants)[i], useCounts);
                updateElement(i, 0, (*invariants)[i], empty_nb,res); 
                // if (atomsSettingBits) {
                //     (*atomsSettingBits)[bit].push_back(std::make_pair(i, 0));
                // }
            }
        }
    }
    // std::cout << "calcFingerprint update 0 done" << std::endl;
    boost::dynamic_bitset<> chiralAtoms(nAtoms);

    // these are the neighborhoods that have already been added
    // to the fingerprint
    std::vector<boost::dynamic_bitset<>> neighborhoods;
    // these are the environments around each atom:
    std::vector<boost::dynamic_bitset<>> atomNeighborhoods(
            nAtoms, boost::dynamic_bitset<>(mol.getNumBonds()));
    boost::dynamic_bitset<> deadAtoms(nAtoms);

    boost::dynamic_bitset<> includeAtoms(nAtoms);
    if (fromAtoms) {
        for (auto idx : *fromAtoms) {
            includeAtoms.set(idx, 1);
        }
    } else {
        includeAtoms.set();
    }

    std::vector<uint> atomOrder(nAtoms);
    if (onlyNonzeroInvariants) {
        std::vector<std::pair<int32_t, uint32_t>> ordering;
        for (uint i = 0; i < nAtoms; ++i) {
            if (!(*invariants)[i]) {
                ordering.emplace_back(1, i);
            } else {
                ordering.emplace_back(0, i);
            }
        }
        std::sort(ordering.begin(), ordering.end());
        for (uint i = 0; i < nAtoms; ++i) {
            atomOrder[i] = ordering[i].second;
        }
    } else {
        for (uint i = 0; i < nAtoms; ++i) {
            atomOrder[i] = i;
        }
    }
    // now do our subsequent rounds:
    for (uint layer = 0; layer < radius; ++layer) {
        // std::cout << "layer: " << layer << std::endl;
        std::vector<uint32_t> roundInvariants(nAtoms);
        std::vector<boost::dynamic_bitset<>> roundAtomNeighborhoods =
                atomNeighborhoods;
        std::vector<MyAccumTuple> neighborhoodsThisRound;

        for (auto atomIdx : atomOrder) {
            // std::cout << "atomIdx: " << atomIdx << std::endl;
            if (!deadAtoms[atomIdx]) {
                // std::cout << "not dead atomIdx: " << atomIdx << std::endl;
                const Atom *tAtom = lmol->getAtomWithIdx(atomIdx);
                if (!tAtom->getDegree()) {
                    deadAtoms.set(atomIdx, 1);
                    continue;
                }
                std::vector<std::pair<int32_t, uint32_t>> nbrs;
                ROMol::OEDGE_ITER beg, end;
                boost::tie(beg, end) = lmol->getAtomBonds(tAtom);
                while (beg != end) {
                    const Bond *bond = mol[*beg];
                    roundAtomNeighborhoods[atomIdx][bond->getIdx()] = 1;

                    uint oIdx = bond->getOtherAtomIdx(atomIdx);
                    roundAtomNeighborhoods[atomIdx] |= atomNeighborhoods[oIdx];

                    int32_t bt = 1;
                    if (useBondTypes) {
                        if (!useChirality || bond->getBondType() != Bond::DOUBLE ||
                                bond->getStereo() == Bond::STEREONONE) {
                            bt = static_cast<int32_t>(bond->getBondType());
                        } else {
                            const int32_t stereoOffset = 100;
                            const int32_t bondTypeOffset = 10;
                            bt = stereoOffset +
                                     bondTypeOffset * static_cast<int32_t>(bond->getBondType()) +
                                     static_cast<int32_t>(bond->getStereo());
                        }
                    }
                    nbrs.emplace_back(bt, (*invariants)[oIdx]);

                    ++beg;
                }

                // sort the neighbor list:
                std::sort(nbrs.begin(), nbrs.end());
                // and now calculate the new invariant and test if the atom is newly
                // "chiral"
                std::uint32_t invar = layer;
                // cout << "invar: " << invar << endl;
                gboost::hash_combine(invar, (*invariants)[atomIdx]);
                // cout << "invar: " << invar << "," << (*invariants)[atomIdx] << endl;
                bool looksChiral = (tAtom->getChiralTag() != Atom::CHI_UNSPECIFIED);
                for (std::vector<std::pair<int32_t, uint32_t>>::const_iterator it =
                                 nbrs.begin();
                         it != nbrs.end(); ++it) {
                    // add the contribution to the new invariant:
                    gboost::hash_combine(invar, *it);
                    // cout << "invar: " << invar << "," << (it->first) << "," << (it->second) << endl;
                    // std::cerr<<"     "<<atomIdx<<": "<<it->first<<" "<<it->second<<" ->
                    // "<<invar<<std::endl;

                    // update our "chirality":
                    if (useChirality && looksChiral && chiralAtoms[atomIdx]) {
                        if (it->first != static_cast<int32_t>(Bond::SINGLE)) {
                            looksChiral = false;
                        } else if (it != nbrs.begin() && it->second == (it - 1)->second) {
                            looksChiral = false;
                        }
                    }
                }
                // cout << "----------------------" << endl;
                if (useChirality && looksChiral) {
                    chiralAtoms[atomIdx] = 1;
                    // add an extra value to the invariant to reflect chirality:
                    std::string cip = "";
                    tAtom->getPropIfPresent(common_properties::_CIPCode, cip);
                    if (cip == "R") {
                        gboost::hash_combine(invar, 3);
                    } else if (cip == "S") {
                        gboost::hash_combine(invar, 2);
                    } else {
                        gboost::hash_combine(invar, 1);
                    }
                }
                roundInvariants[atomIdx] = static_cast<uint32_t>(invar);
                // neighborhoodsThisRound.push_back(boost::make_tuple(roundAtomNeighborhoods[atomIdx],static_cast<uint32_t>(invar), atomIdx));
                neighborhoodsThisRound.push_back(MyAccumTuple(roundAtomNeighborhoods[atomIdx],static_cast<uint32_t>(invar), atomIdx));

                if (!includeRedundantEnvironments &&
                        std::find(neighborhoods.begin(), neighborhoods.end(),
                                            roundAtomNeighborhoods[atomIdx]) != neighborhoods.end()) {
                    // we have seen this exact environment before, this atom
                    // is now out of consideration:
                    deadAtoms[atomIdx] = 1;
                    std::cout <<"p1   atom: "<< atomIdx <<" is dead."<<std::endl;
                    std::cout << "neighborhoods: ";
                    for (auto &n : neighborhoods) {
                        cout << n << ", ";
                    }
                    cout << endl;
                }
            }
        }
        std::sort(neighborhoodsThisRound.begin(), neighborhoodsThisRound.end());
        // std::cout << "neighborhoodsThisRound: " << neighborhoodsThisRound.size() << std::endl;
        int debug_id = 0;
        for (std::vector<MyAccumTuple>::const_iterator iter =
                         neighborhoodsThisRound.begin();
                 iter != neighborhoodsThisRound.end(); ++iter) {
            // if we haven't seen this exact environment before, update the
            // fingerprint:
            // std::cout << debug_id << ", p1" << std::endl;
            if (includeRedundantEnvironments ||
                    std::find(neighborhoods.begin(), neighborhoods.end(),
                                        iter->get<0>()) == neighborhoods.end()) {
                // std::cout << debug_id << ", p2" << std::endl;
                if (!onlyNonzeroInvariants || invariantCpy[iter->get<2>()]) {
                    // std::cout << debug_id << ", p3" << std::endl;
                    if (includeAtoms[iter->get<2>()]) {
                        // std::cout << debug_id << ", p4" << std::endl;
                        // uint32_t bit = updateElement(res, iter->get<1>(), useCounts);
                        updateElement(iter->get<2>(), layer + 1, iter->get<1>(), iter->get<0>(),res);
                        // std::cout << debug_id << ", p5" << std::endl;
                        // if (atomsSettingBits) {
                        //     (*atomsSettingBits)[bit].push_back(
                        //             std::make_pair(iter->get<2>(), layer + 1));
                        // }
                    }
                    if (!fromAtoms || fromAtoms->find(iter->get<2>()) != fromAtoms->end()) {
                        neighborhoods.push_back(iter->get<0>());
                    }
                }
                // std::cerr<<" layer: "<<layer<<" atom: "<<iter->get<2>()<<" "
                // <<iter->get<0>()<< " " << iter->get<1>() << " " <<
                // deadAtoms[iter->get<2>()]<<std::endl;
            } else {
                // we have seen this exact environment before, this atom
                // is now out of consideration:
                std::cout<<"p2   atom: "<< iter->get<2>()<<" is dead."<<std::endl;
                deadAtoms[iter->get<2>()] = 1;
            }
            debug_id++;
        }

        // the invariants from this round become the global invariants:
        std::copy(roundInvariants.begin(), roundInvariants.end(),
                            invariants->begin());

        atomNeighborhoods = roundAtomNeighborhoods;
    }

    if (owner) {
        delete invariants;
    }
}

void IncrementalMorganFingerprint::calcIncrementalFingerprint(const ROMol &mol, const std::list<int> &fromAtoms, Table_T &res) {

    using namespace RDKit::MorganFingerprints;
    #ifdef CENV_DEBUG
    cout << "atom1_idx: " << atom1_idx << endl;
    cout << "atom2_idx: " << atom2_idx << endl;
    cout << "radius: " << this->radius << endl;
    cout << "length: " << this->length << endl;
    #endif
    // I assume that the new added atom and bond are at the last, and other atoms (index) are not changed, but didn't verify this. -- Huanyi  

    const ROMol *lmol = &mol;
    uint nAtoms = mol.getNumAtoms();
    uint nBonds = mol.getNumBonds();
    boost::dynamic_bitset<> empty_nb(nBonds);
    bool includeRingMembership = true;
    gboost::hash<std::vector<uint32_t>> vectHasher;
    std::vector<int> queue;
    std::vector<int> total;
    boost::dynamic_bitset<> checked(nAtoms);
    auto check_atom = [&, this](const int atom_idx) {
        if (atom_idx >= 0) {
            Atom const *atom = mol.getAtomWithIdx(atom_idx);
            std::vector<uint32_t> components;
            components.push_back(atom->getAtomicNum());
            components.push_back(atom->getTotalDegree());
            components.push_back(atom->getTotalNumHs(true));
            components.push_back(atom->getFormalCharge());
            int deltaMass = static_cast<int>(
                atom->getMass() -
                PeriodicTable::getTable()->getAtomicWeight(atom->getAtomicNum()));
            components.push_back(deltaMass);

            if (includeRingMembership &&
                atom->getOwningMol().getRingInfo()->numAtomRings(atom->getIdx())) {
              components.push_back(1);
            }
            uint32_t invar = vectHasher(components);
            // res[0][atom_idx].second = invar;
            updateElement(atom_idx, 0, invar, empty_nb, res);
            queue.push_back(atom_idx);
            total.push_back(atom_idx);
            checked[atom_idx] = 1;
        }
    };

    for (auto aid : fromAtoms) {
        check_atom(aid);
    }

    // add all atoms where bondtype changed
    // The bondtypes are used to decide the order of hashing. 
    // No need to update those atoms' initial hash value.
    {
        // may optimize it.
        int l = std::min<int>(nBonds, this->baseMol.getNumBonds());
        for (int i = 0; i < l; ++i) {
            auto nb = mol.getBondWithIdx(i);
            auto ob = baseMol.getBondWithIdx(i);
            if (nb->getBondType() != ob->getBondType()) {
                queue.push_back(nb->getBeginAtomIdx());
                queue.push_back(nb->getEndAtomIdx());
            }
        }
    }
    sort(queue.begin(), queue.end());
    auto ed = std::unique(queue.begin(), queue.end());
    queue.resize(std::distance(queue.begin(), ed));

    for (uint layer = 0; layer < radius; ++layer) {
        std::vector<int> next_queue;
        for (auto atomIdx : queue) {
            const Atom *tAtom = lmol->getAtomWithIdx(atomIdx);
            ROMol::OEDGE_ITER beg, end;
            boost::tie(beg, end) = lmol->getAtomBonds(tAtom);
            while (beg != end) {
                const Bond *bond = mol[*beg];
                uint oIdx = bond->getOtherAtomIdx(atomIdx);
                if (!checked[oIdx]) {
                    next_queue.push_back(oIdx);
                    total.push_back(oIdx);
                    checked[oIdx] = 1;
                }
                ++beg;
            }       
        }
        for (auto atomIdx : total) {
            const Atom *tAtom = lmol->getAtomWithIdx(atomIdx);
            std::vector<std::pair<int32_t, uint32_t>> nbrs;
            boost::dynamic_bitset<> bs(nBonds);
            ROMol::OEDGE_ITER beg, end;
            boost::tie(beg, end) = lmol->getAtomBonds(tAtom);
            while (beg != end) {
                const Bond *bond = mol[*beg];
                bs[bond->getIdx()] = 1;
                uint oIdx = bond->getOtherAtomIdx(atomIdx);
                bs |= res[layer][oIdx].first;
                int32_t bt = 1;
                if (!useChirality || bond->getBondType() != Bond::DOUBLE ||
                        bond->getStereo() == Bond::STEREONONE) {
                    bt = static_cast<int32_t>(bond->getBondType());
                } else {
                    const int32_t stereoOffset = 100;
                    const int32_t bondTypeOffset = 10;
                    bt = stereoOffset +
                             bondTypeOffset * static_cast<int32_t>(bond->getBondType()) +
                             static_cast<int32_t>(bond->getStereo());
                }
                nbrs.emplace_back(bt, res[layer][oIdx].second);
                ++beg;
            }
            std::sort(nbrs.begin(), nbrs.end());
            std::uint32_t invar = layer;
            // cout << "layer: " << layer << ", atomIdx: " << atomIdx << endl;
            // cout << "invar: " << invar << endl;
            gboost::hash_combine(invar, res[layer][atomIdx].second);
            // cout << "invar: " << invar << "," << res[layer][atomIdx].second << endl;
            for (auto &nb : nbrs) {
                gboost::hash_combine(invar, nb);
                // cout << "invar: " << invar << "," << (nb.first) << "," << (nb.second) << endl;
            }
            updateElement(atomIdx, layer + 1, invar, bs, res);
            // cout << "----------------------" << endl;
        }

        queue = std::move(next_queue);
        // cout << "layer " << layer << "queue: ";
        // for (auto i : queue) {
        //     cout << i << ",";
        // }
        // cout << endl;
    }
}



uint32_t IncrementalMorganFingerprint::updateElement(uint atomIdx, uint radius, uint32_t element, const boost::dynamic_bitset<> &env, Table_T &hash_value_table) {
    hash_value_table[radius][atomIdx] = std::pair<boost::dynamic_bitset<>, uint32_t>(env, element);
    // cout << "updateElement: " << atomIdx << "," << radius << "," << element << "," << env << endl;
    return element;
}
