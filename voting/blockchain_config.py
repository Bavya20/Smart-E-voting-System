from web3 import Web3
import json
import os
from django.conf import settings

# Connect to local Ethereum node
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))  # Change if needed

# Load contract data
with open(os.path.join(settings.BASE_DIR, "contract_data.json")) as f:
    contract_data = json.load(f)

contract_address = Web3.to_checksum_address(contract_data["address"])
contract_abi = contract_data["abi"]

# Initialize contract
contract = web3.eth.contract(address=contract_address, abi=contract_abi)
