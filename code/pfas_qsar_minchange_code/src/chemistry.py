from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception as exc:  # pragma: no cover
    Chem = None
    AllChem = None
    Descriptors = None
    MurckoScaffold = None


ALLOWED_ELEMENTS: Set[str] = {
    "C", "H", "O", "N", "F", "Cl", "Br", "I", "S", "P", "Si", "As", "Hg", "Sn"
}


@dataclass
class StandardizationResult:
    smiles: Optional[str]
    inchikey: Optional[str]
    exclusion_reasons: list[str]


def _require_rdkit() -> None:
    if Chem is None:
        raise ImportError("RDKit is required for chemistry utilities but could not be imported.")


def neutralize_mol(mol):
    _require_rdkit()
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass
    return mol


def standardize_smiles(smiles: str) -> StandardizationResult:
    _require_rdkit()
    reasons: list[str] = []
    if smiles is None or not str(smiles).strip():
        return StandardizationResult(None, None, ["missing_smiles"])

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return StandardizationResult(None, None, ["invalid_smiles"])

    mol = neutralize_mol(mol)
    Chem.RemoveStereochemistry(mol)

    if len(Chem.GetMolFrags(mol)) > 1:
        reasons.append("multiple_fragments")

    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    disallowed = sorted(elements.difference(ALLOWED_ELEMENTS))
    if disallowed:
        reasons.append(f"disallowed_elements:{'|'.join(disallowed)}")

    canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    inchikey = None
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        pass
    return StandardizationResult(canon, inchikey, reasons)


def murcko_scaffold_from_smiles(smiles: str) -> str:
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or MurckoScaffold is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    return scaffold or ""


def contains_disallowed_elements(smiles: str) -> bool:
    res = standardize_smiles(smiles)
    return any(x.startswith("disallowed_elements") for x in res.exclusion_reasons)


def has_aliphatic_c_f(smiles: str) -> bool:
    """
    Proxy for the Martin et al. 'fluorinated local' layer:
    returns True if there is at least one fluorine attached to an aliphatic carbon.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        atoms = (a1, a2)
        syms = {a.GetSymbol() for a in atoms}
        if syms == {"C", "F"}:
            carbon = a1 if a1.GetSymbol() == "C" else a2
            if not carbon.GetIsAromatic() and carbon.GetHybridization().name != "SP2":
                return True
    return False


def morgan_count_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    _require_rdkit()
    if AllChem is None:
        raise ImportError("RDKit AllChem module is not available.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot featurize invalid SMILES: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def molecular_weight_from_smiles(smiles: str) -> Optional[float]:
    _require_rdkit()
    if Descriptors is None:
        raise ImportError("RDKit Descriptors module is not available.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))
