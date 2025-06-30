from web3 import Web3
import json
import os

# Load contract_data.json
with open(os.path.join(os.path.dirname(__file__), 'contract_data.json')) as file:
    contract_data = json.load(file)

# Connect to blockchain (e.g. Ganache)
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))

# Verify connection
if not web3.is_connected():
    raise Exception("Web3 is not connected! Please check your blockchain provider.")

# Extract ABI and contract address
abi = contract_data['abi']
contract_address = Web3.to_checksum_address(contract_data['address'])

# Load contract
contract = web3.eth.contract(address=contract_address, abi=abi)
