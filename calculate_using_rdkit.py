import rdkit
import pandas as pd
from rdkit import Chem
import time
from tqdm import tqdm
import re
all_present_atoms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 30, 33, 34, 35, 36, 37, 38, 47, 52, 53, 54, 55, 56, 83, 88]


atoms_dict = {
    "b": 5,
    "c": 6,
    "n": 7,
    "o": 8,
    "p": 15,
    "s": 16,
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
}
ATOM_PATTERN = "|".join(re.escape(atom_symbol) for atom_symbol in atoms_dict)



def get_all_atoms_rdkit(smile):
    mol = Chem.MolFromSmiles(smile)
    try: 
        return [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    except AttributeError:
        print(f"Encountered error during {smile} smile.")
        return False

def get_all_atoms_string(smiles):
    # Initialize an empty list to store the existing atoms
    existing_atoms = []

    # Iterate over each character in the SMILES string
    i = 0
    while i < len(smiles):
        char = smiles[i]

        # Check if the character is alphabetic
        if char.isalpha():
            # Check if the next character also represents an atom symbol
            if i + 1 < len(smiles) and smiles[i+1].isalpha():
                # Extract the two-letter atom symbol
                atom_symbol = char + smiles[i+1]


                # Check if it's a two-letter atom symbol
                if atom_symbol in atoms_dict:
                    atomic_number = atoms_dict[atom_symbol]
                else:
                    # Assume it's a one-letter atom symbol
                    atomic_number = atoms_dict[char]
            else:
                # Extract the atomic number based on the element symbol
                atomic_number = atoms_dict[char]

            # Add the atomic number to the list if it's not already present
            if atomic_number not in existing_atoms:
                existing_atoms.append(atomic_number)

        i += 1  # Move to the next character


    return existing_atoms

def get_all_unique_elements(smiles):
    two_letter_atoms = {atom_symbol: atom_no for atom_symbol, atom_no in atoms_dict.items() if len(atom_symbol)==2}
    one_letter_atoms = {atom_symbol: atom_no for atom_symbol, atom_no in atoms_dict.items() if len(atom_symbol)==1}

    # first find and delete all the two lettered atoms from smiles string and add those atoms in a set
    # second do this for one letter atoms 
    two_letter_pattern = "|".join(re.escape(atom_symbol) for atom_symbol in two_letter_atoms)
    two_letter_matches = re.findall(two_letter_pattern, smiles)

    # Find and extract all one-letter atoms from the SMILES string
    one_letter_pattern = "|".join(re.escape(atom_symbol) for atom_symbol in one_letter_atoms)
    one_letter_matches = re.findall(one_letter_pattern, smiles)

    # Combine the matches and convert them to unique element numbers
    all_matches = two_letter_matches + one_letter_matches
    return {
        two_letter_atoms[match] if len(match) == 2 else one_letter_atoms[match]
        for match in all_matches
    }

def get_all_unique_elements2(smiles):
    atom_matches = re.findall(ATOM_PATTERN, smiles)
    return {atoms_dict[match] for match in atom_matches}


def get_atom_matrix(df, function):
    # df = df.copy()
    for atom_number in all_present_atoms:
        column_name = f"atomno_{atom_number}"
        df[column_name] = 0

    beginning = time.time()
    print("Calculation started using:", function.__name__)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        smiles = row["canonical_smiles"]
        try:
            existing_atoms = function(smiles)
        except Exception as e:
            print(f"The function {function.__name__} caused an error during this smiles: {smiles}")
            raise e
            

        if existing_atoms is False:
            continue

        for atom_number in existing_atoms:
            column_name = f"atomno_{atom_number}"
            df.at[index, column_name] = 1
    
    runtime = time.time()- beginning
    minutes, seconds = divmod(runtime, 60)
    print(f"Runtime of {function.__name__} is: {int(minutes)}:{seconds:.2f}")
    return df


df = pd.read_csv("chembl_train.smi")
df_matrix = get_atom_matrix(df, get_all_atoms_rdkit)
# df_matrix = get_atom_matrix(df, get_all_unique_elements2)
# print(get_all_unique_elements(df["canonical_smiles"][0]))
# df_matrix.to_csv("smiles_matrix_rdkit.csv", index=False)
# df_matrix.to_csv("smiles_matrix_string2.csv", index=False)
df_matrix.to_csv("druggen_train_rdkit.csv", index=False)
