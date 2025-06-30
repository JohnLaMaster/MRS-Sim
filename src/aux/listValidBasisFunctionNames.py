# listStandardBasisFunctionNames.m
# Georg Oeltzschner, Johns Hopkins University 2022
#
# USAGE:
# allMets = listStandardBasisFunctionNames
#
# DESCRIPTION:
# This function returns a cell array of standard metabolite and
# macromolecular basis function names. It will look up the list of VALID
# names via 'listValidBasisFunctionNames.m'. For metabolites with multiple
# valid names, it will only return the first (standard) name.
#
# OUTPUTS:
# allMets = Cell array with a list of standard metabolite/MM names according
#           to Osprey naming convention.
#
# INPUTS
# type    = String (can be 'mets' or 'mm')
# 
# function listOfStandardNames = listStandardBasisFunctionNames(type)

def listStandardBasisFunctionNames(type_in: str='mets'):
    # Get the list of valid names
    listOfStandardNames = listValidBasisFunctionNames(type_in);

    # Loop over elements and remove all non-default names
    for rr in range(len(listOfStandardNames)):
        if isinstance(listOfStandardNames[rr], list):
            # Pick first element (which is the default name)
            listOfStandardNames[rr] = listOfStandardNames[rr][1];
    
    return listOfStandardNames    



# listValidBasisFunctionNames.m
# Georg Oeltzschner, Johns Hopkins University 2022
#
# USAGE:
# allMets = listValidBasisFunctionNames
# 
# DESCRIPTION:
# This function keeps a record of all valid metabolite names.
# If there is more than one common name for a metabolite, they are grouped
# inside another cell array. 
# The first name inside such a cell array is the default name for that
# metabolite in Osprey.
# 
# OUTPUTS:
# allMets = Cell array with a list of valid metabolite names.
#
# INPUTS
# type    = String (can be 'mets' or 'mm')
# 
# function listOfValidNames = listValidBasisFunctionNames(type)

def listValidBasisFunctionNames(type_in: str='mets'):
    if 'mets' in type_in:
        listOfValidNames =  [
            'AcAc',  # Acetoacetate
            ['Ace', 'Act'],   # Acetate
            ['AcO', 'Acn'],   # Acetone
            'Ala',   # Alanine
            'Asc',   # Ascorbate
            'Asp',   # Aspartate
            'Bet',   # Betaine
            ['bHB', 'bHb'],   # beta-hydroxybutyrate
            ['bHG', '2HG', '2-HG'],   # 2-hydroxyglutarate
            'Car',   # Carnitine
            'Cit',   # Citrate
            ['Cr', 'Cre'],    # Cr
            'Cys',   # Cysteic acid
            'Cystat',# Cystat
            'CrCH2', # negative CrCH2 correction signal
            'EA',    # Ethanolamine
            ['EtOH', 'Eth'],  # Ethanol
            ['fCho', 'Cho'],  # free choline
            'Fuc',   # Fucose
            'GABA',  # GABA
            'Gcn',   # Glucone
            'Gcr',   # Glucoronic acid
            'GPC',   # Glycerophosphocholine
            'GSH',   # Glutathione (reduced)
            'Glc',   # Glucose
            'Gln',   # Glutamine
            'Glu',   # Glutamate
            ['Gly', 'Glyc'],   # Glycine
            'Gua',   # Guanidinoacetate
            'H2O',   # H2O
            'HCar',  # Homocarnosine
            'ILc',   # Isoleucine
            ['mI', 'Ins', 'mIns'],    # myo-inositol
            'Lac',   # Lactate
            'Leu',   # Leucine
            'Lys',   # Lysine
            'NAA',   # N-Acetylaspartate
            'NAAG',  # N-Acetylaspartylglutamate
            ['PCh', 'PCho'],   # Phosphocholine
            'PCr',   # Phosphocreatine
            'PE',    # Phosphoethanolamine
            'Pgc',   # Propyleneglycol
            ['Phenyl', 'PAl'],    # Phenylalanine
            'Pyr',   # Pyruvate
            ['sI', 'Scyllo', 'sIns'],    # scyllo-inositol
            'Ser',   # Serine
            'Suc',   # Succinate
            'Tau',   # Taurine
            'Thr',   # Threonine
            'Tyros', # Tyrosine
            'Val',   # Valine
            'NAA_Ace',   # NAA acetyl
            'NAA_Asp',   # NAA aspartyl
        ]
        
    elif 'mm' in type_in:
        listOfValidNames = [
            'MM09',
            'MM12',
            'MM14',
            'MM17',
            'MM20',
            'MM22',
            'MM27',
            'MM30',
            'MM32',
            'Lip09',
            'Lip13',
            'Lip20',
            'MM37',
            'MM38',
            'MM40',
            'MM42',
            ['MMexp','Mac', 'MMmeas'], # Typical names for measured MM
            'MM_PRESS_PCC',
            'MM_PRESS_CSO',
        ]
            
    return listOfValidNames

# fit_createMetabList.m
# Georg Oeltzschner, Johns Hopkins University 2019.
#
# USAGE:
# metabList = fit_createMetabList;
# 
# DESCRIPTION:
# Creates a list of metabolite basis functions that are to be included in 
# a fit.
# 
# OUTPUTS:
# metabList = structure including flags (1 = included, 0 = excluded) for
#             each metabolite included in the FID-A spin system definition,
#             plus various MM basis functions that may have been included
#             with fit_makeBasis.
#
# INPUTS:
# NONE
# 
# function metabList = fit_createMetabList(includeMetabs)

def fit_createMetabList(includeMetabs):
    # Define the set of available metabolites
    all_mets = listStandardBasisFunctionNames('mets')
    all_mm = listStandardBasisFunctionNames('mm')
    all_mets.extend(all_mm)

    metabList = {}
    for metab in all_mets:
        metabList[metab] = 0

    # Select metabolites to include in basis set depending on user input
    # If 'default' or 'full' are input, fill appropriately...
    if len(includeMetabs) == 1:
        if includeMetabs[0].lower() == 'default':
            # Define the default set
            defaultMets = ['Asc', 'Asp', 'Cr', 'CrCH2', 'GABA', 'GPC', 'GSH', 'Gln', 'Glu',
                           'mI', 'Lac', 'NAA', 'NAAG', 'PCh', 'PCr', 'PE', 'sI', 'Tau',
                           'MM09', 'MM12', 'MM14', 'MM17', 'MM20', 'Lip09', 'Lip13', 'Lip20']

            for metab in defaultMets:
                metabList[metab] = 1
        elif includeMetabs[0].lower() == 'full':
            # Define the full set
            for metab in all_mets:
                metabList[metab] = 1
    else:
        # ... otherwise, if a list of metabolite names is provided, use it.
        for metab in includeMetabs:
            metabList[metab] = 1

    return metabList


# fit_selectMetabs.m
# Georg Oeltzschner, Johns Hopkins University 2019.
#
# USAGE:
# basisSetOut = fit_selectMetabs(basisSetIn, metabList, fitMM)
# 
# DESCRIPTION:
# This function loads the basis set functions for the metabolites
# that have a positive flag set in the structure metabList, which has
# previously been created with fit_createMetabList.
#
# Only basis functions for the metabolites selected in fit_createMetabList
# will be included. This function will output a modified FID-A basis
# set container, which will be subsequently passed on to the fitting
# algorithm.
#
# The function will also include MM and lipid spectra, if present in
# the basis set, and if they were selected to be included.
#
# The other basis functions will be discarded.
# 
# OUTPUTS:
# basisSetOut = FID-A basis set container that only contains the basis
#               functions specified in metabList
#
# INPUTS:
# basisSetIn  = FID-A basis set container (created with fit_makeBasis).
# metabList   = Structure (created with fit_createMetabList) containing
#               flags for each metabolite/MM/lipid basis function that is 
#               supposed to be included in the basis set.
# fitMM       = Flag determining whether MM/lipid basis functions are being
#               kept in the output basis set. 1 = Yes (default), 0 = No.
# 
# function basisSetOut = fit_selectMetabs(basisSetIn, metabList, fitMM)
def fit_selectMetabs(basisSetIn, metabList, fitMM=1):
    # Save all available metabolite names in a list
    all_mets = listStandardBasisFunctionNames('mets')

    # Duplicate the input basis set
    basisSetOut = basisSetIn.copy()

    # Check which metabolites are available in the basis set
    metsInBasisSet = basisSetIn['name']
    metsToKeep = [metab for metab in metsInBasisSet if metab in all_mets]

    # Check for each remaining metabolite in the basis set whether it should be included
    idx_toKeep = [1 if metabList.get(metab, False) else 0 for metab in metsToKeep]
    basisSetOut['nMets'] = sum(idx_toKeep)

    # If the flag for including MM/lipid basis functions is set, include them
    all_MMs = listStandardBasisFunctionNames('mm')

    # Check which of these are available in the basis set
    MMsInBasisSet = basisSetIn['name']
    MMsToKeep = [metab for metab in MMsInBasisSet if metab in all_MMs]
    idx_toKeepMM = [1 if fitMM and metabList.get(metab, False) else 0 for metab in MMsToKeep]

    idx_toKeep.extend(idx_toKeepMM)
    basisSetOut['nMM'] = sum(idx_toKeepMM)

    # Remove the metabolites and MM/lipid basis functions based on the flag values
    basisSetOut['name'] = [metab for i, metab in enumerate(basisSetOut['name']) if idx_toKeep[i]]
    basisSetOut['fids'] = basisSetOut['fids'][:, [i for i, v in enumerate(idx_toKeep) if v], :]
    basisSetOut['specs'] = basisSetOut['specs'][:, [i for i, v in enumerate(idx_toKeep) if v], :]

    return basisSetOut
