
/*

This file implement an incremental method to calculate the morgan fingerprints.

*/


#pragma once

#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/Fingerprints/FingerprintUtil.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <GraphMol/Fingerprints/FingerprintGenerator.h>
#include <GraphMol/Fingerprints/MorganGenerator.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <RDGeneral/BoostEndInclude.h>
#include <RDGeneral/BoostStartInclude.h>
#include <RDGeneral/Exceptions.h>
#include <RDGeneral/hash/hash.hpp>

#include <algorithm>
#include <boost/bind/bind.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/tuple.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <boost/python/numpy.hpp>
#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_3kcompat.h>

using uint = uint;
using Map_T = std::vector<std::pair<boost::dynamic_bitset<>, uint32_t>>;
using Table_T = std::vector<Map_T>;

// Cached fingerprint state for restoring setBaseMol results without recomputation
struct IMFCache {
    Table_T hash_value_table;
    Table_T bond_addition_hvt;
    Table_T atom_addition_hvt;
    RDKit::RWMol baseMol;
    std::vector<uint32_t> invariants;
};

class IncrementalMorganFingerprint {

public:

    uint radius = 2;
    uint length = 2048;

    bool useChirality = 0;
    bool useBondTypes = 1;
    bool onlyNonzeroInvariants = 0;
    bool includeRedundantEnvironments = 1;

    RDKit::FingerprintGenerator<std::uint32_t> *fpGenerator = nullptr;
    RDKit::RWMol baseMol;

    IncrementalMorganFingerprint();
    IncrementalMorganFingerprint(uint radius, uint length);
    void setBaseMol(const RDKit::ROMol &mol);

    // Cache/restore setBaseMol state (cheap memcpy vs expensive recomputation)
    IMFCache cacheState() const;
    void restoreState(const IMFCache &cache);

    void setFingerprintRadius(uint radius) {
        this->radius = radius;
        fpGenerator = RDKit::MorganFingerprint::getMorganGenerator<std::uint32_t>(radius);
        hash_value_table.resize(radius + 1, Map_T());
    }

    void setFingerprintLength(uint length) {
        this->length = length;
    }

    boost::python::list getBaseMolMorganFingerprint();
    boost::python::object getBaseMolMorganFingerprintAsNumPy();

    boost::python::list getIncrementalMorganFingerprint(const RDKit::ROMol &new_mol, const std::list<int> &fromAtoms);
    boost::python::object getIncrementalMorganFingerprintAsNumPy(const RDKit::ROMol &new_mol, const std::list<int> &fromAtoms);
    
    boost::python::list getMorganFingerprint(const RDKit::ROMol &new_mol);
    boost::python::object getMorganFingerprintAsNumPy(const RDKit::ROMol &new_mol);
    boost::python::list getMorganFingerprintByGenerator(const RDKit::ROMol &new_mol);
    boost::python::object getMorganFingerprintAsNumPyByGenerator(const RDKit::ROMol &new_mol);

    // Pure C++ dense FP methods (no Python objects, safe to call without GIL)
    std::vector<uint8_t> getIncrementalMorganFingerprintDense(const RDKit::ROMol &new_mol, const std::list<int> &fromAtoms);
    std::vector<uint8_t> getBaseMolMorganFingerprintDense();
    std::vector<uint8_t> getMorganFingerprintDense(const RDKit::ROMol &new_mol);

private:

    void calcFingerprint(const RDKit::ROMol &mol, uint radius,
        std::vector<uint32_t> *invariants,
        const std::set<uint32_t> *fromAtoms, bool useChirality,
        bool useBondTypes,
        bool onlyNonzeroInvariants,
        bool includeRedundantEnvironments,
        Table_T &);

    void calcBaseFingerprint();
    void calcIncrementalFingerprint(const RDKit::ROMol &new_mol, const std::list<int> &fromAtoms, Table_T &);

    std::set<uint32_t> convertTableToSet(Table_T &);
    boost::python::list convertTableToSparseList(Table_T &);
    boost::python::object convertTableToNumPy(Table_T &);
    std::vector<uint8_t> convertTableToDenseVector(Table_T &);

    uint32_t updateElement(uint atomIdx, uint radius, uint32_t element, const boost::dynamic_bitset<> &,Table_T &res);
    Table_T hash_value_table;
    Table_T atom_addition_hvt;
    Table_T bond_addition_hvt;
    std::vector<uint32_t> invariants;
};