/*

This file re-implement the get_valid_actions in environment.py

*/

#include "environment.h"
#include <boost/python/numpy.hpp>
#include <omp.h>

// Defined in incremental_morgan_fingerprint.cpp — initializes boost::python::numpy
extern void initialize_numpy();

using namespace RDKit;
using std::cout;
using std::endl;

// #define CENV_DEBUG
// #define EMBED_DEBUG

using BondType = RDKit::Bond::BondType;
const static std::array<BondType, 4> bond_order = {
    BondType::UNSPECIFIED,
    BondType::SINGLE,
    BondType::DOUBLE,
    BondType::TRIPLE
};

static std::array<Bond, 4> bonds = {
    Bond(BondType::UNSPECIFIED),
    Bond(BondType::SINGLE),
    Bond(BondType::DOUBLE),
    Bond(BondType::TRIPLE)
};

bool is_in_bond_order(BondType bt) {
    return bt != BondType::UNSPECIFIED && bt <= bond_order[bond_order.size() - 1];
}



Flags::Flags(const std::string &pickle_str) {
    // pickle is a binary string;
    if (pickle_str.length() == 7) {
        int i = 0;
        allow_removal = (pickle_str[i++] == '1');
        allow_remove_entire_bond = (pickle_str[i++] == '1');
        allow_no_modification = (pickle_str[i++] == '1');
        allow_bonds_between_rings = (pickle_str[i++] == '1');
        record_path = (pickle_str[i++] == '1');
        maintain_label = (pickle_str[i++] == '1');
        // old_remove_method = (pickle_str[i++] == '1');
        // maintain_OH = (pickle_str[i++] == '1');
    }
}

std::string Flags::get_pickle() const {
    std::string pickle_str;
    pickle_str.resize(7);
    int i = 0;
    pickle_str[i++] = (allow_removal)?('1'):'0';
    pickle_str[i++] = (allow_remove_entire_bond)?('1'):'0';
    pickle_str[i++] = (allow_no_modification)?('1'):'0';
    pickle_str[i++] = (allow_bonds_between_rings)?('1'):'0';
    pickle_str[i++] = (record_path)?('1'):'0';
    pickle_str[i++] = (maintain_label)?('1'):'0';
    // pickle_str[i++] = (old_remove_method)?('1'):'0';
    // pickle_str[i++] = (maintain_OH)?('1'):'0';
    return pickle_str;
}



MolOps::SanitizeFlags sanitizeMol(ROMol &mol, boost::uint64_t sanitizeOps,
                                  bool catchErrors) {
  auto &wmol = static_cast<RWMol &>(mol);
  uint operationThatFailed;
  if (catchErrors) {
    try {
      MolOps::sanitizeMol(wmol, operationThatFailed, sanitizeOps);
    } catch (const MolSanitizeException &) {
      // this really should not be necessary, but at some point it
      // started to be required with VC++17. Doesn't seem like it does
      // any harm.
    } catch (...) {
    }
  } else {
    MolOps::sanitizeMol(wmol, operationThatFailed, sanitizeOps);
  }
  return static_cast<MolOps::SanitizeFlags>(operationThatFailed);
}


static std::vector<int> max_valences(256, -1); // the maximum valences for atoms

int calMaxValence(int atomic_num) {
    if (atomic_num < 0 || atomic_num >= 256) {
        // errror: ivalid atomic_num
        return -1;
    }
    return max_valences[atomic_num];
};


Environment::Environment(
    const boost::python::list &py_atom_types_str,
    const boost::python::list &py_allowed_ring_sizes,
    const int &morgan_fingerprint_radius, 
    const int &morgan_fingerprint_length,
    const Flags &_flags
    )
{
    std::vector<std::string> atom_types_str {
        boost::python::stl_input_iterator<std::string>(py_atom_types_str),
        boost::python::stl_input_iterator<std::string>()
    };
    
    for (auto &atom_str : atom_types_str) {
        int atomic_num = PeriodicTable::getTable()->getAtomicNumber(atom_str);
        this->atom_types.push_back(atomic_num);
        auto &valences_list = PeriodicTable::getTable()->getValenceList(atomic_num);
        if (valences_list.size() > 0 && atomic_num > 0) {
            max_valences[atomic_num] = *std::max_element(valences_list.begin(), valences_list.end());
        } else {
            isGood = false;
        }
    }

    std::unordered_set<int> allowed_ring_sizes {
        boost::python::stl_input_iterator<int>(py_allowed_ring_sizes),
        boost::python::stl_input_iterator<int>()
    };

    this->allowed_ring_sizes = allowed_ring_sizes;
    this->imf.setFingerprintRadius(morgan_fingerprint_radius);
    this->imf.setFingerprintLength(morgan_fingerprint_length);
    this->flags = flags;
}

Environment::Environment(const std::vector<int> &_at, const std::unordered_set<int> &_al, const int &morgan_fingerprint_radius, const int &morgan_fingerprint_length, const Flags &_f)
{
    this->atom_types = _at;
    this->allowed_ring_sizes = _al;
    this->flags = flags;
    this->imf.setFingerprintRadius(morgan_fingerprint_radius);
    this->imf.setFingerprintLength(morgan_fingerprint_length);
}

Environment::Environment(const Environment &e) 
{
    this->atom_types = e.atom_types;
    this->allowed_ring_sizes = e.allowed_ring_sizes;
    this->isGood = e.isGood;
    this->flags = e.flags;
    this->imf = e.imf;
}

Environment::Environment(const std::string &pickle_str) 
{
    std::stringstream ss(pickle_str);
    int sz;
    ss >> sz;

    int val;
    while (sz-- > 0 && ss >> val) {
        if (val >= 1 && val < 256)
            atom_types.push_back(val);
        else {
            // error
        }
    }

    ss >> sz;
    while (sz-- > 0 && ss >> val) {
        if (val > 0)
            allowed_ring_sizes.insert(val);
        else {
            // error
        }
    }

    uint morgan_fingerprint_radius;
    uint morgan_fingerprint_length;
    ss >> morgan_fingerprint_radius >> morgan_fingerprint_length;

    this->imf.setFingerprintRadius(morgan_fingerprint_radius);
    this->imf.setFingerprintLength(morgan_fingerprint_length);

    std::string flags;
    ss >> flags;
    this->flags = Flags(flags);
}

std::string Environment::get_pickle() const 
{
    std::stringstream ss;
    ss << atom_types.size() << " ";
    for (auto &val : atom_types) ss << val << " ";
    ss << endl;
    ss << allowed_ring_sizes.size() << " ";
    for (auto &val : allowed_ring_sizes) ss << val << " ";
    ss << endl;
    ss << this->imf.radius << " " << this->imf.length << endl;
    ss << flags.get_pickle() << endl;
    return ss.str();
}

boost::python::tuple Environment::get_valid_actions_and_fingerprint(boost::shared_ptr<ROMol> molecule, int get_morgan_fingerprint, int maintain_OH)
{
    /*
        get_morgan_fingerprint: 
            0: do not generate inc fp
            1: generate fp as list
            2: generate fp as numpy

        maintail_OH:
            -2: no limitation
            -1: at least 1 OH bond
            0 ~ N: has the number of OH bonds, it is the same as the initial mol

    */

    boost::shared_ptr<ROMol> mol(new ROMol(*molecule));
    if (mol == nullptr) 
        return {};

    if (get_morgan_fingerprint) {
        imf.setBaseMol(*mol);
    }

    boost::python::list valid_actions;
    boost::python::list fingerprints;

    atom_addition(mol, get_morgan_fingerprint, maintain_OH, valid_actions, fingerprints);
    bond_addition(mol, get_morgan_fingerprint, maintain_OH, valid_actions, fingerprints);
    bond_replace(mol, get_morgan_fingerprint, maintain_OH, valid_actions, fingerprints);
    
    if (this->flags.allow_no_modification) {
        boost::shared_ptr<ROMol> act(new ROMol(*mol));
        valid_actions.append(act);
        if (get_morgan_fingerprint == 1) {
            auto fp = imf.getBaseMolMorganFingerprint();
            fingerprints.append(fp); 
        } else if (get_morgan_fingerprint == 2) {
            auto fp = imf.getBaseMolMorganFingerprintAsNumPy();
            fingerprints.append(fp); 
        }
    }
    return boost::python::make_tuple(valid_actions, fingerprints);

}
boost::python::tuple Environment::get_valid_actions_and_fingerprint_smile(std::string smiles, int get_morgan_fingerprint, int maintain_OH)
{
    /*
        this func uses smiles instead of ROMol. It is slow but has better compatibility.
        get_morgan_fingerprint: 
            0: do not generate inc fp
            1: generate fp as list
            2: generate fp as numpy

        maintail_OH:
            -2: no limitation
            -1: at least 1 OH bond
            0 ~ N: has the number of OH bonds, it is the same as the initial mol

    */
    boost::shared_ptr<ROMol> mol = RDKit::v2::SmilesParse::MolFromSmiles(smiles);
    if (mol == nullptr) 
        return {};

    if (get_morgan_fingerprint) {
        imf.setBaseMol(*mol);
    }
    boost::python::list valid_actions;
    boost::python::list fingerprints;

    atom_addition(mol, get_morgan_fingerprint, maintain_OH, valid_actions, fingerprints);
    bond_addition(mol, get_morgan_fingerprint, maintain_OH, valid_actions, fingerprints);
    bond_replace(mol, get_morgan_fingerprint, maintain_OH, valid_actions, fingerprints);
    if (this->flags.allow_no_modification) {
        boost::shared_ptr<ROMol> act(new ROMol(*mol));
        valid_actions.append(act);
        if (get_morgan_fingerprint == 1) {
            auto fp = imf.getBaseMolMorganFingerprint();
            fingerprints.append(fp); 
        } else if (get_morgan_fingerprint == 2) {
            auto fp = imf.getBaseMolMorganFingerprintAsNumPy();
            fingerprints.append(fp); 
        }
    }
    boost::python::list valid_actions_smile;
    int len = boost::python::len(valid_actions);
    for (int i = 0; i < len; ++i) {
        auto act = boost::python::extract<boost::shared_ptr<ROMol>>(valid_actions[i])();
        valid_actions_smile.append(MolToSmiles(*act));
    }
    return boost::python::make_tuple(valid_actions_smile, fingerprints);

}

void Environment::atom_addition(boost::shared_ptr<ROMol> mol, int get_morgan_fingerprint, int maintain_OH, boost::python::list &valid_actions, boost::python::list &fingerprints) 
{

    if (!mol->getRingInfo()->isInitialized()) {
        MolOps::findSSSR(*mol);
    }


    for (auto atom : mol->atoms()) {
        int atomic_num = atom->getAtomicNum();
        int atomic_idx = atom->getIdx();
        int free_valences = atom->getNumImplicitHs();
        int l = std::min<int>(free_valences, bond_order.size() - 1);
        for (int new_atomic_num : atom_types) {
            if (new_atomic_num != atomic_num || (atomic_num == 6)) { // 6 for 'C'
                int max_valences = calMaxValence(new_atomic_num);
                int max_bond_valences = std::min<int>(l, max_valences);
                for (int i = 1; i <= max_bond_valences; ++i) { // it starts from BondType::SINGLE.
                    RWMol act(*mol);
                    int new_atomic_idx = act.addAtom(new Atom(new_atomic_num), true, true); // bool updateLabel = true, bool takeOwnership = false
                    act.addBond(atomic_idx, new_atomic_idx, bond_order[i]);
                    auto sanitization_result = sanitizeMol(act, MolOps::SANITIZE_ALL, true); // catchErrors = true;
                    if (sanitization_result) {
                        continue;
                    }
                    boost::shared_ptr<ROMol> rt(new ROMol(act, true));
                    if (check_OH(maintain_OH, rt)) {
                        valid_actions.append(rt);
                        if (get_morgan_fingerprint == 1) {
                            auto fp = imf.getIncrementalMorganFingerprint(*rt, {atomic_idx, new_atomic_idx});
                            fingerprints.append(fp);
                        } else if (get_morgan_fingerprint == 2) {
                            auto fp = imf.getIncrementalMorganFingerprintAsNumPy(*rt, {atomic_idx, new_atomic_idx});
                            fingerprints.append(fp);
                        }
                    }                    
                }
            }
        }
    }

}

int Environment::getMinFreeHs(Atom *atom1, Atom *atom2) 
{
    return std::min(atom1->getNumImplicitHs(), atom2->getNumImplicitHs());
}
int Environment::getMinFreeHs(Bond *bond) 
{
    auto atom1 = bond->getBeginAtom();
    auto atom2 = bond->getEndAtom();
    return getMinFreeHs(atom1, atom2);
}

void Environment::bond_addition(boost::shared_ptr<ROMol> mol, int get_morgan_fingerprint, int maintain_OH, boost::python::list &valid_actions, boost::python::list &fingerprints) 
{ 
    // add a new bond between two atoms

    const static int N_ATOMIC_NUM = Atom("N").getAtomicNum();
    const static int O_ATOMIC_NUM = Atom("O").getAtomicNum();

    if (!mol->getRingInfo()->isInitialized()) {
        MolOps::findSSSR(*mol);
    }

    RWMol working_state = RWMol(*mol);
    MolOps::Kekulize(working_state, true); // clearAromaticFlags = true

    auto num_of_atoms = mol->getNumAtoms();
    for (int i = 0; i < num_of_atoms; ++i) {
        auto atom1 = mol->getAtomWithIdx(i);
        bool is_I_in_ring = mol->getRingInfo()->numAtomRings(i);
        for (int j = i + 1; j < num_of_atoms; ++j) {
            bool is_J_in_ring = mol->getRingInfo()->numAtomRings(j);
            std::list<int> shortest_path {i, j};
            auto atom2 = mol->getAtomWithIdx(j);
            auto bond = mol->getBondBetweenAtoms(i, j);
            if (bond) {
                continue;   
            } else if (atom1->getAtomicNum() == atom2->getAtomicNum() && (atom1->getAtomicNum() == N_ATOMIC_NUM || atom1->getAtomicNum() == O_ATOMIC_NUM)) {
                continue;
            } else if (!flags.allow_bonds_between_rings && is_I_in_ring && is_J_in_ring) {
                continue;
            } else if (allowed_ring_sizes.size() > 0) {
                shortest_path = MolOps::getShortestPath(*mol, i, j);
                if (allowed_ring_sizes.find(shortest_path.size()) == allowed_ring_sizes.end()) {
                    continue;
                }
            }
            int l = getMinFreeHs(atom1, atom2);
            l = std::min<int>(l, bond_order.size());
            for (int bid = 1; bid <= l; ++bid) {
                RWMol act(working_state);
                act.addBond(i, j, bond_order[bid]);
                auto sanitization_result = sanitizeMol(act, MolOps::SANITIZE_ALL, true); // catchErrors = true;
                if (sanitization_result) {
                    continue;
                }
                boost::shared_ptr<ROMol> rt(new ROMol(act, true));
                // if ((!flags.maintain_OH) or has_OH(rt)) {
                if (check_OH(maintain_OH, rt)) {
                    valid_actions.append(rt);
                    // if (get_morgan_fingerprint) {
                    //     auto fp = imf.getIncrementalMorganFingerprintAsNumPy(*rt, shortest_path);
                    //     // auto fp = imf.getIncrementalMorganFingerprintAsNumPy(*rt, {i, j});
                    //     fingerprints.append(fp);
                    // } 
                    if (get_morgan_fingerprint == 1) {
                        auto fp = imf.getIncrementalMorganFingerprint(*rt, shortest_path);
                        fingerprints.append(fp);
                    } else if (get_morgan_fingerprint == 2) {
                        auto fp = imf.getIncrementalMorganFingerprintAsNumPy(*rt, shortest_path);
                        fingerprints.append(fp);
                    }               
                }
                
            }
        }
    }
}

void Environment::bond_replace(boost::shared_ptr<ROMol> mol, int get_morgan_fingerprint, int maintain_OH, boost::python::list &valid_actions, boost::python::list &fingerprints) 
{ 
    // remove the entire bond or downgrade it

    const static int N_ATOMIC_NUM = Atom("N").getAtomicNum();
    const static int O_ATOMIC_NUM = Atom("O").getAtomicNum();

    if (!mol->getRingInfo()->isInitialized()) {
        MolOps::findSSSR(*mol);
    }

    RWMol working_state = RWMol(*mol);
    MolOps::Kekulize(working_state, true); // clearAromaticFlags = true
    for (auto bond : mol->bonds()) {
        auto bt = bond->getBondType();
        if (!is_in_bond_order(bt)) continue; // Skip aromatic bonds.

        if (flags.allow_remove_entire_bond) {
            RWMol act(working_state);
            act.removeBond(bond->getBeginAtomIdx(), bond->getEndAtomIdx());
            uint sanitization_result = 0;
            MolOps::sanitizeMol(act, sanitization_result, MolOps::SANITIZE_ALL);
            if (!sanitization_result) {
                auto frags = MolOps::getMolFrags(act, true); // sanitize = true                   
                if (frags.size() > 1) {
                    if (frags[0]->getNumAtoms() < frags[1]->getNumAtoms()) {
                        swap(frags[0], frags[1]);
                    }
                }
                boost::shared_ptr<ROMol> rt(new ROMol(*frags[0], true));
                // if ((!flags.maintain_OH) or has_OH(rt)) {
                if (check_OH(maintain_OH, rt)) {
                    valid_actions.append(rt);
                    // if (get_morgan_fingerprint) {
                    //     int i = bond->getBeginAtomIdx();
                    //     int j = bond->getEndAtomIdx();
                    //     // auto fp = imf.getIncrementalMorganFingerprintAsNumPy(*rt, {i, j}); // to do implement the inc fp for removed bond
                    //     auto fp = imf.getMorganFingerprintAsNumPy(*rt);
                    //     fingerprints.append(fp);    
                    // }
                    if (get_morgan_fingerprint == 1) {
                        auto fp = imf.getMorganFingerprint(*rt);
                        fingerprints.append(fp);
                    } else if (get_morgan_fingerprint == 2) {
                        auto fp = imf.getMorganFingerprintAsNumPy(*rt);
                        fingerprints.append(fp);
                    } 
                }
                
            }
        }

        for (size_t i = 1; i < bond_order.size(); ++i) // skip the BondType::UNSPECIFIED,
        {
            auto new_bond_type = bond_order[i];
            if (new_bond_type != bond->getBondType()) {
                bool canReplace = new_bond_type < bt; // downgrade
                if (new_bond_type > bt) { // upgrade
                    
                    if (new_bond_type - bt <= getMinFreeHs(bond)) {
                        canReplace = true;
                    } else {
                        // safe to break the loop now, because valence is increasing.
                        break;
                    }
                    auto atom1 = bond->getBeginAtom();
                    auto atom2 = bond->getEndAtom();
                    if (atom1->getAtomicNum() == atom2->getAtomicNum() && (atom1->getAtomicNum() == N_ATOMIC_NUM || atom1->getAtomicNum() == O_ATOMIC_NUM)) {
                        // no N=N and O=O. 
                        // MolDQN did forbid this, I don't know why N=N is not allowed. --Huanyi
                        canReplace = false;
                    }
                }

                if (canReplace) {
                    RWMol act = RWMol(working_state);
                    act.getBondWithIdx(bond->getIdx())->setBondType(bond_order[i]);
                    auto sanitization_result = sanitizeMol(act, MolOps::SANITIZE_ALL, true); // catchErrors = true;
                    if (sanitization_result) {
                        continue;
                    }
                    boost::shared_ptr<ROMol> rt(new ROMol(act, true));
                    // if ((!flags.maintain_OH) or has_OH(rt)) {
                    if (check_OH(maintain_OH, rt)) {
                        valid_actions.append(rt);
                        // if (get_morgan_fingerprint) {
                        //     int i = bond->getBeginAtomIdx();
                        //     int j = bond->getEndAtomIdx();
                        //     auto fp = imf.getIncrementalMorganFingerprintAsNumPy(*rt, {i, j});
                        //     fingerprints.append(fp);    
                        // }

                        if (get_morgan_fingerprint == 1) {
                            int i = bond->getBeginAtomIdx();
                            int j = bond->getEndAtomIdx();
                            auto fp = imf.getIncrementalMorganFingerprint(*rt, {i, j});
                            fingerprints.append(fp);
                        } else if (get_morgan_fingerprint == 2) {
                            int i = bond->getBeginAtomIdx();
                            int j = bond->getEndAtomIdx();
                            auto fp = imf.getIncrementalMorganFingerprintAsNumPy(*rt, {i, j});
                            fingerprints.append(fp);    
                        } 

                    }
                }
            }
        }
    }
}

int Environment::count_OH(boost::shared_ptr<RDKit::ROMol> mol) {
    int OH_count = 0;
    for (auto atom : mol->atoms()) {
        if (atom->getAtomicNum() == 8 && atom->getNumImplicitHs() > 0) { // 8 for 'O'
            OH_count ++;
        }
    }
    return OH_count;
}

bool Environment::check_OH(int maintain_OH, boost::shared_ptr<RDKit::ROMol> mol) {
    // -2: no limitation
    // -1: at least 1 OH bond
    // 0 ~ N: the mol should have the number of OH bonds, it is the same as the initial mol

    if (maintain_OH == -2) {
        return true;
    }
    int coh = count_OH(mol);
    if (maintain_OH == -1) {
        if (coh >= 1) return true;
    } else if (maintain_OH >= 0) {
        if (coh == maintain_OH) return true;
    }

    return false;

}

// ---------------------------------------------------------------------------
// Internal methods for batch valid action computation (no Python objects)
// These mirror atom_addition/bond_addition/bond_replace but use C++ containers
// and dense fingerprint vectors instead of boost::python::list.
// ---------------------------------------------------------------------------

void Environment::atom_addition_internal(boost::shared_ptr<ROMol> mol, int maintain_OH, BatchActionResult &result)
{
    if (!mol->getRingInfo()->isInitialized()) {
        MolOps::findSSSR(*mol);
    }

    for (auto atom : mol->atoms()) {
        int atomic_num = atom->getAtomicNum();
        int atomic_idx = atom->getIdx();
        int free_valences = atom->getNumImplicitHs();
        int l = std::min<int>(free_valences, bond_order.size() - 1);
        for (int new_atomic_num : atom_types) {
            if (new_atomic_num != atomic_num || (atomic_num == 6)) {
                int max_vals = calMaxValence(new_atomic_num);
                int max_bond_valences = std::min<int>(l, max_vals);
                for (int i = 1; i <= max_bond_valences; ++i) {
                    RWMol act(*mol);
                    int new_atomic_idx = act.addAtom(new Atom(new_atomic_num), true, true);
                    act.addBond(atomic_idx, new_atomic_idx, bond_order[i]);
                    auto sanitization_result = sanitizeMol(act, MolOps::SANITIZE_ALL, true);
                    if (sanitization_result) continue;
                    boost::shared_ptr<ROMol> rt(new ROMol(act, true));
                    if (check_OH(maintain_OH, rt)) {
                        result.actions.push_back(rt);
                        result.fingerprints.push_back(
                            imf.getIncrementalMorganFingerprintDense(*rt, {atomic_idx, new_atomic_idx}));
                    }
                }
            }
        }
    }
}

void Environment::bond_addition_internal(boost::shared_ptr<ROMol> mol, int maintain_OH, BatchActionResult &result)
{
    const static int N_ATOMIC_NUM = Atom("N").getAtomicNum();
    const static int O_ATOMIC_NUM = Atom("O").getAtomicNum();

    if (!mol->getRingInfo()->isInitialized()) {
        MolOps::findSSSR(*mol);
    }

    RWMol working_state = RWMol(*mol);
    MolOps::Kekulize(working_state, true);

    auto num_of_atoms = mol->getNumAtoms();
    for (int i = 0; i < (int)num_of_atoms; ++i) {
        auto atom1 = mol->getAtomWithIdx(i);
        bool is_I_in_ring = mol->getRingInfo()->numAtomRings(i);
        for (int j = i + 1; j < (int)num_of_atoms; ++j) {
            bool is_J_in_ring = mol->getRingInfo()->numAtomRings(j);
            std::list<int> shortest_path {i, j};
            auto atom2 = mol->getAtomWithIdx(j);
            auto bond = mol->getBondBetweenAtoms(i, j);
            if (bond) {
                continue;
            } else if (atom1->getAtomicNum() == atom2->getAtomicNum() && (atom1->getAtomicNum() == N_ATOMIC_NUM || atom1->getAtomicNum() == O_ATOMIC_NUM)) {
                continue;
            } else if (!flags.allow_bonds_between_rings && is_I_in_ring && is_J_in_ring) {
                continue;
            } else if (allowed_ring_sizes.size() > 0) {
                shortest_path = MolOps::getShortestPath(*mol, i, j);
                if (allowed_ring_sizes.find(shortest_path.size()) == allowed_ring_sizes.end()) {
                    continue;
                }
            }
            int l = getMinFreeHs(atom1, atom2);
            l = std::min<int>(l, bond_order.size());
            for (int bid = 1; bid <= l; ++bid) {
                RWMol act(working_state);
                act.addBond(i, j, bond_order[bid]);
                auto sanitization_result = sanitizeMol(act, MolOps::SANITIZE_ALL, true);
                if (sanitization_result) continue;
                boost::shared_ptr<ROMol> rt(new ROMol(act, true));
                if (check_OH(maintain_OH, rt)) {
                    result.actions.push_back(rt);
                    result.fingerprints.push_back(
                        imf.getIncrementalMorganFingerprintDense(*rt, shortest_path));
                }
            }
        }
    }
}

void Environment::bond_replace_internal(boost::shared_ptr<ROMol> mol, int maintain_OH, BatchActionResult &result)
{
    const static int N_ATOMIC_NUM = Atom("N").getAtomicNum();
    const static int O_ATOMIC_NUM = Atom("O").getAtomicNum();

    if (!mol->getRingInfo()->isInitialized()) {
        MolOps::findSSSR(*mol);
    }

    RWMol working_state = RWMol(*mol);
    MolOps::Kekulize(working_state, true);

    for (auto bond : mol->bonds()) {
        auto bt = bond->getBondType();
        if (!is_in_bond_order(bt)) continue;

        if (flags.allow_remove_entire_bond) {
            RWMol act(working_state);
            act.removeBond(bond->getBeginAtomIdx(), bond->getEndAtomIdx());
            uint sanitization_result = 0;
            MolOps::sanitizeMol(act, sanitization_result, MolOps::SANITIZE_ALL);
            if (!sanitization_result) {
                auto frags = MolOps::getMolFrags(act, true);
                if (frags.size() > 1) {
                    if (frags[0]->getNumAtoms() < frags[1]->getNumAtoms()) {
                        swap(frags[0], frags[1]);
                    }
                }
                boost::shared_ptr<ROMol> rt(new ROMol(*frags[0], true));
                if (check_OH(maintain_OH, rt)) {
                    result.actions.push_back(rt);
                    result.fingerprints.push_back(imf.getMorganFingerprintDense(*rt));
                }
            }
        }

        for (size_t i = 1; i < bond_order.size(); ++i) {
            auto new_bond_type = bond_order[i];
            if (new_bond_type != bond->getBondType()) {
                bool canReplace = new_bond_type < bt;
                if (new_bond_type > bt) {
                    if (new_bond_type - bt <= getMinFreeHs(bond)) {
                        canReplace = true;
                    } else {
                        break;
                    }
                    auto atom1 = bond->getBeginAtom();
                    auto atom2 = bond->getEndAtom();
                    if (atom1->getAtomicNum() == atom2->getAtomicNum() && (atom1->getAtomicNum() == N_ATOMIC_NUM || atom1->getAtomicNum() == O_ATOMIC_NUM)) {
                        canReplace = false;
                    }
                }

                if (canReplace) {
                    RWMol act = RWMol(working_state);
                    act.getBondWithIdx(bond->getIdx())->setBondType(bond_order[i]);
                    auto sanitization_result = sanitizeMol(act, MolOps::SANITIZE_ALL, true);
                    if (sanitization_result) continue;
                    boost::shared_ptr<ROMol> rt(new ROMol(act, true));
                    if (check_OH(maintain_OH, rt)) {
                        result.actions.push_back(rt);
                        int bi = bond->getBeginAtomIdx();
                        int bj = bond->getEndAtomIdx();
                        result.fingerprints.push_back(
                            imf.getIncrementalMorganFingerprintDense(*rt, {bi, bj}));
                    }
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// bond_addition_row_internal: Process one row (atom_i) of bond_addition
// mol must have ring info; working_state is pre-kekulized.
// ---------------------------------------------------------------------------
void Environment::bond_addition_row_internal(
    boost::shared_ptr<ROMol> mol,
    const RWMol &working_state,
    int atom_i,
    int maintain_OH,
    BatchActionResult &result)
{
    const static int N_ATOMIC_NUM = Atom("N").getAtomicNum();
    const static int O_ATOMIC_NUM = Atom("O").getAtomicNum();

    auto num_of_atoms = mol->getNumAtoms();
    auto atom1 = mol->getAtomWithIdx(atom_i);
    bool is_I_in_ring = mol->getRingInfo()->numAtomRings(atom_i);

    for (int j = atom_i + 1; j < (int)num_of_atoms; ++j) {
        bool is_J_in_ring = mol->getRingInfo()->numAtomRings(j);
        std::list<int> shortest_path {atom_i, j};
        auto atom2 = mol->getAtomWithIdx(j);
        auto bond = mol->getBondBetweenAtoms(atom_i, j);

        if (bond) {
            continue;
        } else if (atom1->getAtomicNum() == atom2->getAtomicNum() &&
                   (atom1->getAtomicNum() == N_ATOMIC_NUM || atom1->getAtomicNum() == O_ATOMIC_NUM)) {
            continue;
        } else if (!flags.allow_bonds_between_rings && is_I_in_ring && is_J_in_ring) {
            continue;
        } else if (allowed_ring_sizes.size() > 0) {
            shortest_path = MolOps::getShortestPath(*mol, atom_i, j);
            if (allowed_ring_sizes.find(shortest_path.size()) == allowed_ring_sizes.end()) {
                continue;
            }
        }
        int l = getMinFreeHs(atom1, atom2);
        l = std::min<int>(l, bond_order.size());
        for (int bid = 1; bid <= l; ++bid) {
            RWMol act(working_state);
            act.addBond(atom_i, j, bond_order[bid]);
            auto sanitization_result = sanitizeMol(act, MolOps::SANITIZE_ALL, true);
            if (sanitization_result) continue;
            boost::shared_ptr<ROMol> rt(new ROMol(act, true));
            if (check_OH(maintain_OH, rt)) {
                result.actions.push_back(rt);
                result.fingerprints.push_back(
                    imf.getIncrementalMorganFingerprintDense(*rt, shortest_path));
            }
        }
    }
}


// ---------------------------------------------------------------------------
// get_valid_actions_batch: Compute valid actions for multiple molecules
// using OpenMP parallelism. Returns (list_of_action_lists, list_of_fp_arrays)
// where each fp_array is numpy [n_actions, fp_length] uint8.
// ---------------------------------------------------------------------------
boost::python::tuple Environment::get_valid_actions_batch(
    boost::python::list py_molecules,
    boost::python::list py_maintain_OH_flags,
    int nThreads)
{
    namespace bp = boost::python;
    const int n = bp::len(py_molecules);

    initialize_numpy();

    // Phase 1: Extract molecules and flags (GIL held)
    std::vector<boost::shared_ptr<ROMol>> mols(n);
    std::vector<int> maintain_OH(n);
    for (int i = 0; i < n; i++) {
        mols[i] = bp::extract<boost::shared_ptr<ROMol>>(py_molecules[i]);
        maintain_OH[i] = bp::extract<int>(py_maintain_OH_flags[i]);
    }

    // Pre-create thread-local Environment copies (GIL held, safe for cache copy)
    int actual_threads = std::min(nThreads, std::max(1, n));
    std::vector<Environment> thread_envs;
    thread_envs.reserve(actual_threads);
    for (int t = 0; t < actual_threads; t++) {
        thread_envs.emplace_back(*this);
    }

    // Per-molecule results
    std::vector<BatchActionResult> results(n);

    // Phase 2: Compute valid actions in parallel (GIL released)
    // Split into 2a (per-mol: atom_add + bond_replace + precompute) and
    // 2b (flat per-row: bond_addition) for better load balance on large mols.

    // Pre-compute per-molecule data for bond_addition row tasks
    struct MolPrecomp {
        boost::shared_ptr<ROMol> mol;       // with ring info initialized
        RWMol working_state;                // kekulized copy
        IMFCache imf_cache;                 // cached fingerprint state
    };
    std::vector<MolPrecomp> precomp(n);

    Py_BEGIN_ALLOW_THREADS

    // Phase 2a: atom_addition + bond_replace + precompute (per molecule)
    #pragma omp parallel for schedule(dynamic) num_threads(actual_threads)
    for (int i = 0; i < n; i++) {
        int tid = omp_get_thread_num();
        Environment &tenv = thread_envs[tid];

        precomp[i].mol = boost::shared_ptr<ROMol>(new ROMol(*mols[i]));
        if (!precomp[i].mol->getRingInfo()->isInitialized())
            MolOps::findSSSR(*precomp[i].mol);

        // Pre-kekulize working state for bond_addition rows
        precomp[i].working_state = RWMol(*precomp[i].mol);
        MolOps::Kekulize(precomp[i].working_state, true);

        // Set base fingerprint and cache it
        tenv.imf.setBaseMol(*precomp[i].mol);
        precomp[i].imf_cache = tenv.imf.cacheState();

        // atom_addition and bond_replace (O(n) per mol — well balanced)
        tenv.atom_addition_internal(precomp[i].mol, maintain_OH[i], results[i]);
        tenv.bond_replace_internal(precomp[i].mol, maintain_OH[i], results[i]);

        if (tenv.flags.allow_no_modification) {
            boost::shared_ptr<ROMol> act(new ROMol(*precomp[i].mol));
            results[i].actions.push_back(act);
            results[i].fingerprints.push_back(tenv.imf.getBaseMolMorganFingerprintDense());
        }
    }

    // Phase 2b: bond_addition — flat per-row tasks for load balance
    // Each task = one atom row (j > atom_i) of one molecule
    struct RowTask { int mol_idx; int atom_i; };
    std::vector<RowTask> row_tasks;
    row_tasks.reserve(n * 20);  // rough estimate
    for (int i = 0; i < n; i++) {
        int natoms = precomp[i].mol->getNumAtoms();
        for (int a = 0; a < natoms; a++) {
            row_tasks.push_back({i, a});
        }
    }
    int total_row_tasks = (int)row_tasks.size();

    // Per-task results (no contention — each task writes to its own)
    std::vector<BatchActionResult> row_results(total_row_tasks);

    // Use chunk size = avg atoms/mol so consecutive rows of same mol
    // go to same thread, minimizing cache restores.
    int avg_atoms = std::max(1, total_row_tasks / std::max(1, n));
    #pragma omp parallel num_threads(actual_threads)
    {
        int tid = omp_get_thread_num();
        Environment &tenv = thread_envs[tid];
        int last_mol_idx = -1;

        #pragma omp for schedule(dynamic, avg_atoms)
        for (int t = 0; t < total_row_tasks; t++) {
            int mi = row_tasks[t].mol_idx;
            int ai = row_tasks[t].atom_i;

            // Only restore when switching molecules (tasks grouped by mol)
            if (mi != last_mol_idx) {
                tenv.imf.restoreState(precomp[mi].imf_cache);
                last_mol_idx = mi;
            }

            tenv.bond_addition_row_internal(
                precomp[mi].mol, precomp[mi].working_state,
                ai, maintain_OH[mi], row_results[t]);
        }
    }

    // Phase 2c: merge bond_addition results into per-molecule results
    for (int t = 0; t < total_row_tasks; t++) {
        auto &rr = row_results[t];
        if (!rr.actions.empty()) {
            int mi = row_tasks[t].mol_idx;
            results[mi].actions.insert(results[mi].actions.end(),
                std::make_move_iterator(rr.actions.begin()),
                std::make_move_iterator(rr.actions.end()));
            results[mi].fingerprints.insert(results[mi].fingerprints.end(),
                std::make_move_iterator(rr.fingerprints.begin()),
                std::make_move_iterator(rr.fingerprints.end()));
        }
    }

    Py_END_ALLOW_THREADS

    // Phase 3: Convert to Python objects (GIL held)
    namespace np = boost::python::numpy;
    np::dtype dt = np::dtype::get_builtin<uint8_t>();
    uint fp_len = this->imf.length;

    bp::list py_actions_list;
    bp::list py_fps_list;

    for (int i = 0; i < n; i++) {
        bp::list py_actions;
        const auto &result = results[i];
        int n_actions = result.actions.size();

        for (int j = 0; j < n_actions; j++) {
            py_actions.append(result.actions[j]);
        }
        py_actions_list.append(py_actions);

        // Create numpy array [n_actions, fp_len] uint8
        if (n_actions > 0) {
            np::ndarray fps = np::zeros(bp::make_tuple(n_actions, (int)fp_len), dt);
            uint8_t *data = reinterpret_cast<uint8_t *>(fps.get_data());
            for (int j = 0; j < n_actions; j++) {
                std::memcpy(data + j * fp_len, result.fingerprints[j].data(), fp_len);
            }
            py_fps_list.append(fps);
        } else {
            py_fps_list.append(np::zeros(bp::make_tuple(0, (int)fp_len), dt));
        }
    }

    return bp::make_tuple(py_actions_list, py_fps_list);
}


// ---------------------------------------------------------------------------
// embed_molecules_parallel: Parallel ETKDG conformer generation using OpenMP
//
// For each molecule (must already have explicit H via Chem.AddHs):
//   1. Copy molecule to C++ RWMol (GIL held)
//   2. Attempt EmbedMolecule up to maxAttempts times (GIL released, OpenMP parallel)
//   3. Record first successful conformer coords and compute success probability
//
// Returns: (coords_list, success_list, probs_list, natoms_list)
//   coords_list[i]: numpy array [N, 3] of doubles, or None if embedding failed
//   success_list[i]: bool
//   probs_list[i]: float in [0, 1], fraction of successful attempts
//   natoms_list[i]: int, number of atoms (including H)
// ---------------------------------------------------------------------------
boost::python::tuple embed_molecules_parallel(
    boost::python::list py_molecules,
    int maxAttempts,
    int nThreads,
    int maxIterations,
    int timeout)
{
    namespace bp = boost::python;
    const int n = bp::len(py_molecules);

    // Ensure numpy C API is initialized (idempotent)
    initialize_numpy();

    // Phase 1: Extract molecules + precompute natoms (GIL held)
    std::vector<RWMol> mols_h(n);
    std::vector<int> natoms(n, 0);
    for (int i = 0; i < n; i++) {
        boost::shared_ptr<ROMol> mol_ptr = bp::extract<boost::shared_ptr<ROMol>>(py_molecules[i]);
        mols_h[i] = RWMol(*mol_ptr);
        natoms[i] = mols_h[i].getNumAtoms();
    }

    // Output arrays (no Python objects — safe without GIL)
    std::vector<std::vector<double>> all_coords(n);
    std::vector<double> probs(n, 0.0);

    // Atomic flags for flat parallel ETKDG with early exit
    std::vector<std::atomic<int>> first_success(n);
    std::vector<std::atomic<int>> success_counts(n);
    std::vector<std::atomic<int>> actual_attempts(n);
    for (int i = 0; i < n; i++) {
        first_success[i].store(0);
        success_counts[i].store(0);
        actual_attempts[i].store(0);
    }

    // Phase 2: ETKDG — flat task parallelism with early exit (GIL released)
    // Task ordering: attempt-major (all mols attempt 0, then all mols attempt 1, ...)
    // This ensures attempt 0 runs for all mols first; after that, mols that already
    // succeeded can be skipped via early exit — saving up to (maxAttempts-1)/maxAttempts work.
    Py_BEGIN_ALLOW_THREADS

    int total_tasks = n * maxAttempts;
    int actual = std::min(nThreads, total_tasks);

    #pragma omp parallel for schedule(dynamic) num_threads(actual)
    for (int task = 0; task < total_tasks; task++) {
        int i = task % n;           // molecule index (attempt-major order)
        // Early exit: skip if this molecule already has a successful embedding
        if (first_success[i].load(std::memory_order_relaxed)) {
            continue;
        }
        actual_attempts[i].fetch_add(1, std::memory_order_relaxed);
        try {
            RWMol mol_copy(mols_h[i]);
            DGeomHelpers::EmbedParameters params;
            params.useRandomCoords = true;
            params.clearConfs = true;
            params.randomSeed = -1;
            params.maxIterations = maxIterations;
            if (timeout > 0) params.timeout = timeout;
            int cid = DGeomHelpers::EmbedMolecule(mol_copy, params);
            if (cid >= 0) {
                success_counts[i].fetch_add(1, std::memory_order_relaxed);
                int expected = 0;
                if (first_success[i].compare_exchange_strong(expected, 1)) {
                    // Only the first successful thread extracts coordinates
                    const int N = natoms[i];
                    const auto &conf = mol_copy.getConformer(cid);
                    all_coords[i].resize(N * 3);
                    for (int a = 0; a < N; a++) {
                        const auto &pos = conf.getAtomPos(a);
                        all_coords[i][a * 3]     = pos.x;
                        all_coords[i][a * 3 + 1] = pos.y;
                        all_coords[i][a * 3 + 2] = pos.z;
                    }
                }
            }
        } catch (...) {
            // Silently ignore embedding errors
        }
    }

    Py_END_ALLOW_THREADS

    // Phase 3: Convert C++ results to Python objects (GIL held)
    namespace np = boost::python::numpy;
    np::dtype dt = np::dtype::get_builtin<double>();

    bp::list py_coords_list;
    bp::list py_success_list;
    bp::list py_probs_list;
    bp::list py_natoms_list;

    for (int i = 0; i < n; i++) {
        int succ = first_success[i].load();
        int att = actual_attempts[i].load();
        // prob = successes / actual attempts (accurate even with early exit)
        probs[i] = att > 0 ? static_cast<double>(success_counts[i].load()) / att : 0.0;
        py_success_list.append(succ != 0);
        py_probs_list.append(probs[i]);
        py_natoms_list.append(natoms[i]);

        if (succ) {
            np::ndarray arr = np::zeros(bp::make_tuple(natoms[i], 3), dt);
            std::memcpy(arr.get_data(), all_coords[i].data(),
                        natoms[i] * 3 * sizeof(double));
            py_coords_list.append(arr);
        } else {
            py_coords_list.append(bp::object());  // None
        }
    }

    return bp::make_tuple(py_coords_list, py_success_list, py_probs_list, py_natoms_list);
}


using namespace boost::python;

struct Environment_pickle_suit : boost::python::pickle_suite
{
    static boost::python::tuple getinitargs(const Environment& e) {
        return boost::python::make_tuple(e.get_pickle());
    }
};

BOOST_PYTHON_MODULE(cenv)
{
    class_<Flags>("Flags")
        .def_readwrite("allow_removal", &Flags::allow_removal)
        .def_readwrite("allow_remove_entire_bond", &Flags::allow_remove_entire_bond)
        .def_readwrite("allow_no_modification", &Flags::allow_no_modification)
        .def_readwrite("allow_bonds_between_rings", &Flags::allow_bonds_between_rings)
        .def_readwrite("record_path", &Flags::record_path)
        .def_readwrite("maintain_label", &Flags::maintain_label);
        // .def_readwrite("old_remove_method", &Flags::old_remove_method);
        // .def_readwrite("maintain_OH", &Flags::maintain_OH);

    class_<Environment>("Environment", init<boost::python::list, boost::python::list, int, int, Flags>())
        .def(init<std::string>())
        .def("get_valid_actions_and_fingerprint", &Environment::get_valid_actions_and_fingerprint)
        .def("get_valid_actions_and_fingerprint_smile", &Environment::get_valid_actions_and_fingerprint_smile)
        .def("get_valid_actions_batch", &Environment::get_valid_actions_batch,
            (arg("molecules"), arg("maintain_OH_flags"), arg("nThreads")=72))
        .def_pickle(Environment_pickle_suit());

    def("embed_molecules_parallel", &embed_molecules_parallel,
        (arg("molecules"), arg("maxAttempts")=7, arg("nThreads")=72,
         arg("maxIterations")=0, arg("timeout")=1));
}
