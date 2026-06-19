#pragma once
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/Fingerprints/FingerprintUtil.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
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
#include <fstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <unistd.h>

#include "incremental_morgan_fingerprint.h"

struct Flags {
    bool allow_removal = true; // downgrade bond
    bool allow_remove_entire_bond = true;
    bool allow_no_modification = true;
    bool allow_bonds_between_rings = false; 
    bool record_path = false;
    bool maintain_label = false;
    // bool old_remove_method = true;
    // bool maintain_OH = true;

    Flags() = default;
    Flags(const std::string &pickle_str);
    std::string get_pickle() const;

};

class Environment
{
public:

    std::vector<int> atom_types; // the atomicNum for atoms that can be added to molecules in atom_addition
    std::unordered_set<int> allowed_ring_sizes;
    IncrementalMorganFingerprint imf;
    bool isGood = true;
    Flags flags;
    std::unordered_map<std::string, boost::python::tuple> cache;

    Environment(
        const boost::python::list &py_atom_types_str,
        const boost::python::list &py_allowed_ring_sizes,
        const int &morgan_fingerprint_radius, 
        const int &morgan_fingerprint_length,
        const Flags &_flags
        );
    Environment(const std::vector<int> &_at, const std::unordered_set<int> &_al, const int &morgan_fingerprint_radius, const int &morgan_fingerprint_length, const Flags &_f);
    Environment(const Environment &e);
    Environment(const std::string &pickle_str);
    std::string get_pickle() const;
    boost::python::tuple get_valid_actions_and_fingerprint(boost::shared_ptr<RDKit::ROMol> molecule, int get_morgan_fingerprint = 0, int maintain_OH = -1);
    boost::python::tuple get_valid_actions_and_fingerprint_smile(std::string smiles, int get_morgan_fingerprint = 0, int maintain_OH = -1);
    void atom_addition(boost::shared_ptr<RDKit::ROMol> mol, int get_morgan_fingerprint, int maintain_OH, boost::python::list &valid_actions, boost::python::list &fingerprints);
    int getMinFreeHs(RDKit::Atom *atom1, RDKit::Atom *atom2);
    int getMinFreeHs(RDKit::Bond *bond);
    void bond_addition(boost::shared_ptr<RDKit::ROMol> mol, int get_morgan_fingerprint, int maintain_OH, boost::python::list &valid_actions, boost::python::list &fingerprints);
    void bond_replace(boost::shared_ptr<RDKit::ROMol> mol, int get_morgan_fingerprint, int maintain_OH, boost::python::list &valid_actions, boost::python::list &fingerprints);
    int count_OH(boost::shared_ptr<RDKit::ROMol> mol);
    bool check_OH(int maintain_OH, boost::shared_ptr<RDKit::ROMol> mol);
};