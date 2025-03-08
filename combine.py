import pandas as pd
from pymatgen.core import Structure

def main():
    materials = ["high_BN", "high_P", "high_InSe", "high_GaSe", "high_MoS2", "high_WSe2", "low_MoS2", "low_WSe2"]

    # Handle non tmdc dataset
    for i in materials :
        parts = i.split("_")
        the_material = parts[1]

        # Load the data to df
        defects_df = pd.read_csv(f"{i}/defects.csv")
        description_df = pd.read_csv(f"{i}/descriptors.csv")
        structure_df = pd.read_csv(f"{i}/initial_structures.csv")
        elements_df = pd.read_csv(f"{i}/elements.csv")

        
        # Prepare the descriptor df
        # Change the column name of the descriptor id column
        description_df = description_df.rename(columns={"_id": "descriptor_id"})

        # Clearly represent the defects in the description_df
        description_df = description_df.apply(lambda row: string_to_columns(row), axis= 1).fillna(0)

        # Clearly specify the base for future stratification
        description_df["dataset_material"] = i

        # Add description to defects df
        merged_df = defects_df.merge(description_df, on="descriptor_id", how="left")

        # Modify the merged data
        # formation energy 
        merged_df = merged_df.apply(lambda row: get_ef(row, structure_df, elements_df, the_material), axis=1)

        # Energy per atom
        if "high" in i: # Add to high density dataset only
            merged_df = merged_df.apply(lambda row: energy_per_atom(row, i), axis=1)

        # Get definate homo and lumo values and normalize
        # hBN,BP,GaSe,InSe have the majority and minority attribues
        if "2" not in i:
            merged_df = merged_df.apply(remove_majmin, axis= 1)
            merged_df = merged_df.apply(lambda row: normalize(row, structure_df, the_material), axis=1)

        # MoS2 and WSe2 only need to be normalized
        else:
            merged_df = merged_df.apply(lambda row: normalize(row,structure_df, the_material),axis=1)

        # Remove the unrequired columns and add total mag where necessary
        if "2" not in i:
            merged_df = merged_df.drop(["defects", "descriptor_id", "homo_majority", "lumo_majority",
                                        "homo_lumo_gap_majority","E_1_majority", "homo_minority", 
                                        "lumo_minority", "homo_lumo_gap_minority", "E_1_minority",
                                        "homo", "lumo", "description"], axis=1)
            
        elif "2" in i and "high" in i:
            merged_df = merged_df.drop(["defects", "descriptor_id", "homo_lumo_gap", 
                                        "homo", "lumo", "description"], axis=1)
            merged_df["total_mag"] = 0

        elif "2" in i and "low" in i:
            merged_df = merged_df.drop(["defects", "descriptor_id", "homo_lumo_gap", 
                                        "band_gap", "homo", "lumo", "description"], axis=1)
            merged_df["total_mag"] = 0

        # Return the new df as csv
        new_csv_file = f"{i}.csv"
        merged_df.to_csv(new_csv_file)
    
    
def string_to_sites(a_column):
    # Remove unwanted chars
    unwanted_chars = ['[',']']
    for i in unwanted_chars:
        a_column = a_column.replace(i,"")

    # Create a list of the different types of defects
    types = a_column.split("}")
    new_types = [j + "}" for j in types]

    # Remove the additional "{" at the end of the list
    del new_types[-1]

    # Remove the " ," before the "{"
    new_new_types = [types.lstrip(" ,") for types in new_types]

    # Defects clearly represented in 
    list_of_dicts = [eval(dict_string) for dict_string in new_new_types]

    list_of_defects = []
    for i in list_of_dicts:
        if i["type"] == "vacancy":
            defect = f'vacant_{i["element"]}'
            list_of_defects.append(defect)

        elif i["type"] == "substitution":
            defect = f'sub_{i["from"]}_{i["to"]}'
            list_of_defects.append(defect)

        else:
            list_of_defects.append("ubnormal")

    # Create a dictionary of defect_type: number_of_sites
    the_dict = {defect: list_of_defects.count(defect) for defect in list_of_defects}

    return the_dict

def string_to_columns(row):
    dict_defects = string_to_sites(row["defects"])

    for i,j in dict_defects.items():
        row[i] = j

    row.fillna(0.0, inplace=True)
    return row
    

def get_ef(row, structure_df, elements_df, base):
    # Read value of "energy" and save it as Ed
    E_defect = row["energy"]

    # Read value of "energy" in initial structure.csv for material and save as Epristine
    E_pristine = structure_df.loc[structure_df["base"] == base, "energy"].iloc[0]

    # Get the defects in the df
    all_columns = list(row.index)  
    defects_columns = [col for col in all_columns if "vacant" in col or "sub" in col]
    
    # Get defect:site pair
    defects_dict = {i:row[i] for i in defects_columns}
    total_sites = sum(defects_dict.values())

    # Get list of niui(The number of atoms i * chemical potential of atom i)
    list_niui = []
    for x, y in defects_dict.items():
        if "vacant" in x:
            parts = x.split("_")
            vacant_element = parts[1]
            vac_chem_pot = elements_df.loc[elements_df["element"] == vacant_element, "chemical_potential"].iloc[0]
            
            niui_rem = y * (vac_chem_pot * - 1)
            list_niui.append(niui_rem)
        elif "sub" in x:
            parts = x.split("_")
            removed_element = parts[1]
            added_element = parts[2]
            rem_chem_pot = elements_df.loc[elements_df["element"] == removed_element, "chemical_potential"].iloc[0]
            add_chem_pot = elements_df.loc[elements_df["element"] == added_element, "chemical_potential"].iloc[0]

            niui_rem = y * (rem_chem_pot * -1)
            list_niui.append(niui_rem)
            
            niui_add = y * add_chem_pot
            list_niui.append(niui_add)
        else:
            raise ValueError(f"Unrecognized defect type: {x}")

    # Get the sum of niui
    the_sum = sum(list_niui)
    
    # The formation energy
    row["Formation_energy"] = E_defect - E_pristine - the_sum

    # The formation energy per site
    row["Formation_energy_per_site"] = row["Formation_energy"]/total_sites

    return row

def energy_per_atom(row, data_base):
    cif_file = f"{data_base}/cifs/{row["_id"]}.cif"
    structure = Structure.from_file(cif_file)
    sites_no = structure.num_sites

    row["total_sites"] = sites_no
    row["energy_per_atom"] = row["energy"]/ row["total_sites"]

    return row

def remove_majmin(row):
    row["homo"] = (row["homo_majority"] + row["homo_minority"])/2
    row["lumo"] = (row["lumo_majority"] + row["lumo_minority"])/2
    row["E_1"] = (row["E_1_majority"] + row["E_1_minority"])/2

    return row

def normalize(row, structure_df, base):
    E_1_pristine = structure_df.loc[structure_df["base"] == base, "E_1"].iloc[0]
    E_vbm_pristine = structure_df.loc[structure_df["base"] == base, "E_VBM"].iloc[0]

    row["norm_homo"] = row["homo"] - row["E_1"] - (E_vbm_pristine - E_1_pristine)
    row["norm_lumo"] = row["lumo"] - row["E_1"] - (E_vbm_pristine - E_1_pristine)

    return row

if __name__ == "__main__":
    main()