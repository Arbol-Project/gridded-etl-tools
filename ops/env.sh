# This file contains convenience functions for running common IPFS tasks from the CLI. Including it in the shell environment
# is not currently necessary for using the repo. It can be added to the current shell environment by running:
#
#     $ source ./env.sh

# List only the hash values in the IPNS table.
function ipns_published_names
{
    ipfs key list -l | cut -d' ' -f1
}

# Print only the ID fields of the JSON files associated with each IPNS key. This will print the ID of each published STAC.
# It can be used, for example, to verify which dataset metadata's are published.
function ipns_metadata_check
{
    for name in $(ipns_published_names);
    do
        ipfs dag get $(ipfs name resolve "$name") | python3 -m json.tool | grep \"id\";
    done
}

# Print only the name field of the JSON files associated with each recursive pin. This will print the name of every pinned
# Zarr. It can be used, for example, to verify which dataset Zarrs are pinned.
function ipfs_pinned_zarr
{
    for recursive_hash in $(ipfs pin ls -q -t recursive);
    do
        ipfs dag get "$recursive_hash" | python3 -m json.tool | grep \"name\";
    done
}

# Print only the ID field of the JSON files associated with each direct pin. This will print the name of every pinned STAC.
# It can be used, for example, to verify which dataset STAC's are pinned.
function ipfs_pinned_metadata
{
    for direct_hash in $(ipfs pin ls -q -t direct);
    do
        ipfs dag get "$direct_hash" | python3 -m json.tool | grep \"id\";
    done
}

# Print the STAC associated with an IPNS key. Prints to the system pager for convenience.
function ipns_stac
{
    # Check if an argument, the IPNS key, has been provided
    if [ -n "$1" ]
    then
        for ipns_hash in $(ipfs key list -l | grep -i "$1 " | cut -d' ' -f1);
        do
            ipfs dag get $(ipfs name resolve "$ipns_hash") | python3 -m json.tool | less;
        done
    else
        echo "missing required argument: IPNS key"
    fi
}

# Print the IPFS hash associated with the Zarr associated with the given IPNS key.
#
# NOTE: requires the `jq` utility:
#
#     apt install jq
#
function ipns_zarr_hash
{
    # Ensure an argument, the IPNS key, has been provided
    if [ -n "$1" ]
    then
        ipns_hash=$(ipfs key list -l | grep "$1 " | cut -d' ' -f1)
        if [ -n "$ipns_hash" ]
        then
            ipfs dag get $(ipfs name resolve $ipns_hash) | jq -r '.assets.zmetadata.href."/"'
        else
            echo "key not found: $1"
        fi
    else
        echo "missing required argument: IPNS key"
    fi
}
