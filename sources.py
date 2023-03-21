##### sources.py
# 
# This file instatiates all climate data manager sets and groups them into lists that provide an easy way
# to access sources by organization.
#
# When using this file, it makes sense to import it and then access the classes using the `SOURCES` list or 
# some subset of it. This way, classes are only instatiated once, and classes can be processed by iterating
# through the list. 
#
# All instances of a set manager class will compare equal to one another even if they have different ids:
#
#     > import CHIRPS                                                                                                                                
#     > a = CHIRPS.CHIRPSFinal05()                                                                                                                   
#     > b = CHIRPS.CHIRPSFinal05()                                                                                                                   
#     > id(a)                                                                                                                                        
#     140610729850384
#     > id(b)                                                                                                                                        
#     140610084163792
#     > a == b                                                                                                                                       
#     True
#
# One use of this file is to define which sets are available and then check which sets are being processed.
# For example, you could start by presenting the user with the `SOURCES` list. As an option to python's
# argparse library, that looks like:
#
#     parser.add_argument("--source", choices=SOURCES)
#
# You could check if a source is from the CHIRPS organization using:  
#
#     source in CHIRPS_SOURCES
#
# Or, you could use `isinstance` to do the same:
#
#     isinstance(source, CHIRPS.CHIRPS)

import sys, inspect
from etls.managers.chirps import CHIRPSFinal05, CHIRPSFinal25, CHIRPSPrelim05
SOURCES = [tup[1] for tup in inspect.getmembers(sys.modules[__name__], inspect.isclass)]
# use the following to get a set manager object from a string:
#
#     $ get_set_manager_from_name("chirps_final_05")
#     $ -> <Class: CHIRPSFinal05>

def get_set_manager_from_name(name):
    for source in SOURCES:
        if source.name() == name:
            return source
    print(f"failed to set manager from name {name}")

def get_set_manager_from_key(key):
    for source in SOURCES:
        # special handling of GFS which generates multiple keys in a single class
        if source.json_key() == key:
            return source
    print(f"failed to get a manager from key {key}")
