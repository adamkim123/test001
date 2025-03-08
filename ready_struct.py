from pymatgen.core import Structure, PeriodicSite, DummySpecie
from pymatgen.analysis.local_env import MinimumDistanceNN
import numpy as np
import pandas as pd

def main():
    materials = ["high_BN", "high_P", "high_InSe", "high_GaSe", "high_MoS2", "high_WSe2", "low_MoS2", "low_WSe2"]

    for i in materials:
        defects_df = pd.read_csv(f"{i}.csv")
        for j in range(len(defects_df)):
            # Get defective structure
            cif_id = defects_df["_id"][j]
            file_path = f"{i}/cifs/{cif_id}.cif"
            defective_struct = Structure.from_file(file_path)

            # Get reference structure
            ref_unit_cell = Structure.from_file(f"{i}/unit.cif")
            cell_size = defects_df["cell_size"][j]
            reference_struct = ref_unit_cell.make_supercell(cell_size)

            # add defects to defective structure
            new_defective_struct, all_defects, defect_an_property = get_defects(defective_struct, reference_struct)

            # Get properties of each site of the defective structure
            sites_properties = get_properties(new_defective_struct, all_defects)

            # Add defect type and atomic number change property to properties gotten
            for key1 in sites_properties.keys():
                if key1 in defect_an_property.keys():
                    sites_properties[key1].update(defect_an_property[key1])
                else:
                    pass

            # Add the repective properties to each site
            for site_struct in new_defective_struct.sites:
                if site_struct in sites_properties.keys():
                    site_struct.properties.update(sites_properties[site_struct])
                else:
                    pass

            # Add structure wide properties to the structure
            # Get columns
            struct_wide_properties = {}
            def_columns = defects_df.columns
            for x in def_columns:
                struct_wide_properties[x] = defects_df[x][j]

            new_defective_struct.struct_properties = struct_wide_properties


def struct_to_dict(structure):
    list_of_sites = structure.sites
    list_of_frac_coords = np.round(structure.frac_coords,3)
    structure_dict = {i: j for i, j in zip(list_of_sites, list_of_frac_coords)}
    return structure_dict

def get_defects(defective_struct, reference_struct):
    copy_defective_struct = defective_struct.copy()
    # struct to dict
    defective_dict = struct_to_dict(copy_defective_struct)
    reference_dict = struct_to_dict(reference_struct)

    # Get lattice of defective structure
    structure_lattice = copy_defective_struct.lattice

    # Define list to add all defect sites
    defects_list = []

    # Define a dictionary to hold atomic numbers
    site_property = {}

    for ref_site, ref_coords in reference_dict.items():
        matching = False
        for def_site, def_coords in defective_dict.items():
            if np.array_equal(ref_coords, def_coords):
                matching = True
                if ref_site.specie != def_site.specie: # Substitution case
                    # Add site to defects list
                    defects_list.append(def_site)

                    # Get atomic number change and defect type
                    add_property = {"orig_new_an":(ref_site.specie.Z, def_site.specie.Z),
                                    "atomic_number_change": def_site.specie.Z - ref_site.specie.Z,
                                    "vacancy_defect": 0.0,
                                    "substitution_defect": 1.0}
                    site_property[def_site] = add_property
                else:
                    add_property = {"orig_new_an":(ref_site.specie.Z, def_site.specie.Z),
                                    "atomic_number_change": def_site.specie.Z - ref_site.specie.Z,
                                    "vacancy_defect": 0.0,
                                    "substitution_defect": 0.0}
                    site_property[def_site] = add_property

        if not matching: # Vacancy case
            # Add site to defective structure
            vacant_site = PeriodicSite(
                species= DummySpecie(),
                coords= ref_coords,
                coords_are_cartesian= False, 
                lattice= structure_lattice
                )
            
            copy_defective_struct.append(vacant_site.species, vacant_site.frac_coords)

            # Add site to defects list
            defects_list.append(vacant_site)

            # Get atomic number change and defect type
            add_property={"orig_new_an": (ref_site.specie.Z, 0),
                          "atomic_number_change": 0 - ref_site.specie.Z,
                          "vacancy_defect": 1.0,
                          "substitution_defect": 0.0}
            site_property[vacant_site] = add_property

    # Till this point every defect site is in the defective structure
    return copy_defective_struct, defects_list, site_property



def get_properties(new_defective_struct, all_defect_sites):
    list_of_sites = new_defective_struct.sites

    # Define an instance responsible for getting the nearest neighbors
    cnn = MinimumDistanceNN(cutoff=3.0)

    # dictionary of properties
    site_dict = {} # Of each site
    new_dict = {}  # Of all sites

    # Lets get the properties for each defect site
    for i in list_of_sites:
        the_index = list_of_sites.index(i)

        # Get the nearest neighbors of each defect site
        neighbors = cnn.get_nn_info(new_defective_struct, the_index)

        # Get the coordination numbers
        coord_numbers = len(neighbors)

        # Get the vital attributes:
        # neighbors
        the_neighbors = []
        # site index of neighbor
        neighbors_site_index = []
        # neighbors image
        neighbors_image = []
        # distance to neighbors
        bl_to_neighbors = []
        # weights to neighbors
        weights_to_neighbors = []
        # atomic numbers of neighbors
        elements_an = []

        for neighbor_info in neighbors:
            the_neighbors.append(neighbor_info['site'])
            neighbors_site_index.append(neighbor_info['site_index'])
            neighbors_image.append(neighbor_info['image'])
            bond_length = neighbor_info['site'].distance(i)
            bl_to_neighbors.append(bond_length)
            weights_to_neighbors.append(neighbor_info['weight'])
            # Handle the atomic numbers
            the_an = neighbor_info["site"].specie.Z
            if the_an > 0 and the_an < 119:
                elements_an.append(the_an)
            else:
                the_an = 0
                elements_an.append(the_an)
                
        # Get distance to other defect sites
        dist_to_other_defects = []
        for n in all_defect_sites:
            dist = i.distance(n)
            dist_to_other_defects.append(dist)

        # Collect these properties as dictionaries
        site_dict[i] = {"nearest_neighbors": the_neighbors,
                        "neighbors_atomic_numbers": elements_an,
                        "nearest_neighbors_index": neighbors_site_index,
                        "neighbors_image": neighbors_image,
                        "neighbors_weight": weights_to_neighbors,
                        "distance_to_neighbors": bl_to_neighbors,
                        "coordination_number": coord_numbers,
                        "distance_to_defects": dist_to_other_defects}
        
        new_dict.update(site_dict)
    
    return site_dict

if __name__ == "__main__":
    main()