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

// void test(const std::string path) {
// 	std::vector<int> atom_types {6, 8, 7};
// 	std::unordered_set<int> allowed_ring_sizes {3, 5, 6};
// 	const int morgan_fingerprint_radius = 3;
// 	const int morgan_fingerprint_length = 2048;
// 	const Flags f;
// 	Environment e(atom_types, allowed_ring_sizes, morgan_fingerprint_radius, morgan_fingerprint_length, f);
	
// }

int main(int argc, char const *argv[])
{
	// test("");
	using namespace RDKit;
	RDKit::ROMol *mol = RDKit::SmilesToMol( "Cc1ccccc1" );
	int new_atomic_num = 7;
	RWMol act(*mol);
    int new_atomic_idx = act.addAtom(new Atom(new_atomic_num));
	return 0;
}