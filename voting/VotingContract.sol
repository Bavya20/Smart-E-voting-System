// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Voting {
    struct Voter {
        bool voted;
        string vote;
    }

    mapping(address => Voter) public voters;
    mapping(string => uint256) public votes;

    function castVote(string memory party) public {
        require(!voters[msg.sender].voted, "You have already voted.");
        voters[msg.sender] = Voter(true, party);
        votes[party]++;
    }

    function getVotes(string memory party) public view returns (uint256) {
        return votes[party];
    }
}

pragma solidity ^0.8.0;

contract Voting {
    mapping(address => bool) public hasVoted;
    mapping(string => uint256) public votes;

    function vote(string memory candidate) public {
        require(!hasVoted[msg.sender], "You have already voted!");
        votes[candidate] += 1;
        hasVoted[msg.sender] = true;
    }

    function getVotes(string memory candidate) public view returns (uint256) {
        return votes[candidate];
    }
}

