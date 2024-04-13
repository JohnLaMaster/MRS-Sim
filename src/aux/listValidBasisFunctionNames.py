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
